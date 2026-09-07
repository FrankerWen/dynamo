// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! FIFO reuse policy for inactive registered blocks.
//!
//! The queue is intrusive over dense `BlockId`s. This keeps insert, remove,
//! and eviction O(1) without a per-entry `HashMap`/`BTreeMap` node. The only
//! backing storage is a set of parallel vectors that grows to the largest
//! materialized block id.

#![allow(dead_code)]

use super::reuse_policy::*;

use super::{BlockId, InactiveBlock};

/// FIFO reuse policy
#[derive(Debug)]
pub(crate) struct FifoReusePolicy {
    blocks: Vec<Option<InactiveBlock>>,
    previous: Vec<Option<BlockId>>,
    next: Vec<Option<BlockId>>,
    head: Option<BlockId>,
    tail: Option<BlockId>,
    count: usize,
}

impl Default for FifoReusePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl FifoReusePolicy {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            previous: Vec::new(),
            next: Vec::new(),
            head: None,
            tail: None,
            count: 0,
        }
    }

    fn ensure_slot(&mut self, block_id: BlockId) {
        let required = block_id.saturating_add(1);
        if self.blocks.len() < required {
            self.blocks.resize(required, None);
            self.previous.resize(required, None);
            self.next.resize(required, None);
        }
    }

    fn unlink(&mut self, block_id: BlockId) -> Option<InactiveBlock> {
        let block = self.blocks.get_mut(block_id)?.take()?;
        let previous = self.previous[block_id].take();
        let next = self.next[block_id].take();
        match previous {
            Some(previous) => self.next[previous] = next,
            None => self.head = next,
        }
        match next {
            Some(next) => self.previous[next] = previous,
            None => self.tail = previous,
        }
        self.count = self.count.saturating_sub(1);
        Some(block)
    }
}

impl ReusePolicy for FifoReusePolicy {
    fn insert(&mut self, inactive_block: InactiveBlock) -> Result<(), ReusePolicyError> {
        let block_id = inactive_block.block_id;
        self.ensure_slot(block_id);
        assert!(self.blocks[block_id].is_none(), "block already exists");
        self.blocks[block_id] = Some(inactive_block);
        self.previous[block_id] = self.tail;
        self.next[block_id] = None;
        match self.tail {
            Some(tail) => self.next[tail] = Some(block_id),
            None => self.head = Some(block_id),
        }
        self.tail = Some(block_id);
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    fn remove(&mut self, block_id: BlockId) -> Result<(), ReusePolicyError> {
        self.unlink(block_id)
            .map(|_| ())
            .ok_or(ReusePolicyError::BlockNotFound(block_id))
    }

    fn next_free(&mut self) -> Option<InactiveBlock> {
        self.unlink(self.head?)
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::hash_for_tokens;

    /// Helper function to create InactiveBlock instances for testing.
    fn create_inactive_block(block_id: BlockId, seed: u64) -> InactiveBlock {
        InactiveBlock {
            block_id,
            seq_hash: hash_for_tokens(&[seed as u32]),
        }
    }

    #[test]
    fn test_fifo_ordering_basic() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks
        let block1 = create_inactive_block(1, 100);
        let block2 = create_inactive_block(2, 200);
        let block3 = create_inactive_block(3, 300);

        let seq_hash1 = block1.seq_hash;
        let seq_hash2 = block2.seq_hash;
        let seq_hash3 = block3.seq_hash;

        policy.insert(block1).unwrap();
        policy.insert(block2).unwrap();
        policy.insert(block3).unwrap();

        // Verify FIFO order - first inserted should come out first
        assert_eq!(policy.len(), 3);
        assert!(!policy.is_empty());

        let retrieved1 = policy.next_free().unwrap();
        assert_eq!(retrieved1.block_id, 1);
        assert_eq!(retrieved1.seq_hash, seq_hash1);

        let retrieved2 = policy.next_free().unwrap();
        assert_eq!(retrieved2.block_id, 2);
        assert_eq!(retrieved2.seq_hash, seq_hash2);

        let retrieved3 = policy.next_free().unwrap();
        assert_eq!(retrieved3.block_id, 3);
        assert_eq!(retrieved3.seq_hash, seq_hash3);

        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
    }

    #[test]
    fn test_fifo_ordering_with_delays() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks
        let blocks = vec![
            create_inactive_block(10, 1000),
            create_inactive_block(20, 2000),
            create_inactive_block(30, 3000),
            create_inactive_block(40, 4000),
        ];

        for block in blocks {
            policy.insert(block).unwrap();
        }

        // Retrieve all blocks and verify FIFO order
        let expected_order = vec![10, 20, 30, 40];
        let mut retrieved_order = Vec::new();

        while let Some(block) = policy.next_free() {
            retrieved_order.push(block.block_id);
        }

        assert_eq!(retrieved_order, expected_order);
    }

    #[test]
    fn test_insert_and_remove() {
        let mut policy = FifoReusePolicy::new();

        // Insert several blocks
        let blocks = vec![
            create_inactive_block(1, 100),
            create_inactive_block(2, 200),
            create_inactive_block(3, 300),
            create_inactive_block(4, 400),
        ];

        for block in blocks {
            policy.insert(block).unwrap();
        }

        assert_eq!(policy.len(), 4);

        // Remove block 2 (second inserted)
        policy.remove(2).unwrap();
        assert_eq!(policy.len(), 3);

        // Retrieve remaining blocks - should be 1, 3, 4 in that order
        let retrieved1 = policy.next_free().unwrap();
        assert_eq!(retrieved1.block_id, 1);

        let retrieved2 = policy.next_free().unwrap();
        assert_eq!(retrieved2.block_id, 3);

        let retrieved3 = policy.next_free().unwrap();
        assert_eq!(retrieved3.block_id, 4);

        assert!(policy.is_empty());
    }

    #[test]
    fn test_empty_operations() {
        let mut policy = FifoReusePolicy::new();

        // Test empty state
        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
        assert!(policy.next_free().is_none());

        // Insert and remove a block
        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();
        assert!(!policy.is_empty());
        assert_eq!(policy.len(), 1);

        let retrieved = policy.next_free().unwrap();
        assert_eq!(retrieved.block_id, 1);

        // Should be empty again
        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
        assert!(policy.next_free().is_none());
    }

    #[test]
    #[should_panic(expected = "block already exists")]
    fn test_duplicate_block_panic() {
        let mut policy = FifoReusePolicy::new();

        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();

        // Inserting the same block ID again should panic
        let duplicate_block = create_inactive_block(1, 200); // Same ID, different hash
        policy.insert(duplicate_block).unwrap();
    }

    #[test]
    fn test_remove_nonexistent_block() {
        let mut policy = FifoReusePolicy::new();

        // Try to remove from empty policy
        let result = policy.remove(999);
        assert!(matches!(result, Err(ReusePolicyError::BlockNotFound(_))));

        // Insert a block and try to remove a different one
        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();

        let result = policy.remove(999);
        assert!(matches!(result, Err(ReusePolicyError::BlockNotFound(_))));

        // Verify the original block is still there
        assert_eq!(policy.len(), 1);
        let retrieved = policy.next_free().unwrap();
        assert_eq!(retrieved.block_id, 1);
    }

    #[test]
    fn test_interleaved_operations() {
        let mut policy = FifoReusePolicy::new();

        // Insert some blocks
        policy.insert(create_inactive_block(1, 100)).unwrap();
        policy.insert(create_inactive_block(2, 200)).unwrap();
        policy.insert(create_inactive_block(3, 300)).unwrap();

        // Remove the first one
        let first = policy.next_free().unwrap();
        assert_eq!(first.block_id, 1);

        // Insert another block
        policy.insert(create_inactive_block(4, 400)).unwrap();

        // Remove a specific block by ID
        policy.remove(3).unwrap();

        // Insert another block
        policy.insert(create_inactive_block(5, 500)).unwrap();

        // The remaining blocks should come out in order: 2, 4, 5
        let second = policy.next_free().unwrap();
        assert_eq!(second.block_id, 2);

        let third = policy.next_free().unwrap();
        assert_eq!(third.block_id, 4);

        let fourth = policy.next_free().unwrap();
        assert_eq!(fourth.block_id, 5);

        assert!(policy.is_empty());
    }

    #[test]
    fn test_intrusive_link_ordering() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks and verify the dense link vectors encode insertion
        // order without an auxiliary tree or hash table.
        for i in 1..=5 {
            policy
                .insert(create_inactive_block(i, i as u64 * 100))
                .unwrap();
        }
        assert_eq!(policy.head, Some(1));
        assert_eq!(policy.tail, Some(5));
        assert_eq!(policy.previous[1], None);
        assert_eq!(policy.next[1], Some(2));
        assert_eq!(policy.previous[5], Some(4));
        assert_eq!(policy.next[5], None);

        // Retrieve all blocks and verify they come out in insertion order
        let mut retrieval_order: Vec<BlockId> = Vec::new();
        while let Some(block) = policy.next_free() {
            retrieval_order.push(block.block_id);
        }

        let expected_order: Vec<BlockId> = (1..=5).collect();
        assert_eq!(retrieval_order, expected_order);
    }

    #[test]
    fn test_dense_vectors_grow_to_largest_materialized_id() {
        let mut policy = FifoReusePolicy::new();
        policy.insert(create_inactive_block(3, 300)).unwrap();
        policy.insert(create_inactive_block(100, 10_000)).unwrap();
        assert_eq!(policy.blocks.len(), 101);
        assert_eq!(policy.previous.len(), 101);
        assert_eq!(policy.next.len(), 101);
        assert_eq!(policy.next_free().unwrap().block_id, 3);
        assert_eq!(policy.next_free().unwrap().block_id, 100);
    }
}
