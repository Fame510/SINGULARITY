/// RDMA Fabric — GPU-direct memory transfers across nodes.
///
/// STATUS: this crate defines the fabric API only. The RDMA backend is
/// NOT implemented here — operations that would move bytes return an
/// error rather than pretend to succeed. The production transport is
/// the UCX (Unified Communication X) nerve in the proprietary engine,
/// which moves GPU-to-GPU over RoCE v2 with zero copy.
pub struct Fabric {
    /// Device context for RDMA operations
    device: String,
    /// Active queue pairs
    active_qps: usize,
}

impl Fabric {
    pub fn new(device: &str) -> Self {
        Self {
            device: device.to_string(),
            active_qps: 0,
        }
    }

    /// Register a GPU memory region for remote access.
    ///
    /// Not implemented in this crate: returns an error rather than a
    /// fake memory-region key.
    pub fn register_memory(&mut self, vram_ptr: *const u8, size: usize) -> anyhow::Result<u64> {
        let _ = (vram_ptr, size);
        anyhow::bail!(
            "RDMA backend not implemented in singularity-transport \
             (device {}): memory registration requires the UCX nerve",
            self.device
        )
    }

    /// Execute a zero-copy RDMA write: GPU_A → GPU_B.
    ///
    /// Not implemented in this crate: returns an error rather than
    /// reporting a transfer that moved no bytes.
    pub fn teleport_direct(
        &self,
        src_mr: u64,
        dest_node: &str,
        size: usize,
    ) -> anyhow::Result<()> {
        let _ = (src_mr, size);
        anyhow::bail!(
            "RDMA backend not implemented in singularity-transport: \
             teleport to {} requires the UCX nerve",
            dest_node
        )
    }
}
