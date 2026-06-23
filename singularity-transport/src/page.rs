/// Context Page management for SINGULARITY.
///
/// KV-cache is sharded into 16MB "Context Pages" for efficient
/// transfer and management across the cluster.

pub const PAGE_SIZE: usize = 16 * 1024 * 1024; // 16MB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTier {
    /// Active in local VRAM — ~1ns latency
    Hot,
    /// On peer GPU via NVLink/RDMA — ~2-5μs
    Warm,
    /// In host system RAM via PCIe — ~10-20μs
    Moist,
    /// On local NVMe storage — ~100μs
    Cold,
}

#[derive(Debug, Clone)]
pub struct ContextPage {
    pub page_id: u64,
    pub session_id: u64,
    pub tier: PageTier,
    pub size: usize,
    pub local_addr: Option<*const u8>,
    pub remote_node: Option<String>,
}

impl ContextPage {
    pub fn new(page_id: u64, session_id: u64, size: usize) -> Self {
        Self {
            page_id,
            session_id,
            tier: PageTier::Hot,
            size: size.min(PAGE_SIZE),
            local_addr: None,
            remote_node: None,
        }
    }

    /// Promote this page to a faster tier.
    pub fn promote(&mut self, target: PageTier) {
        self.tier = target;
    }
}
