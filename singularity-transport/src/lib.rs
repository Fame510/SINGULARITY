//! SINGULARITY Transport Fabric
//!
//! Open fabric API for RDMA/RoCE v2 zero-copy GPU-to-GPU KV-cache
//! teleportation: page/buffer types and the `Fabric` interface.
//!
//! STATUS: the RDMA backend is not implemented in this crate. Calls
//! that would move bytes fail loudly with an error (never a silent
//! no-op success). The production transport is the UCX nerve in the
//! proprietary engine.
//!
//! This is the OPEN SOURCE component of SINGULARITY.
//! Licensed under Apache 2.0.

pub mod fabric;
pub mod page;
pub mod buffer;

/// Initialize the transport layer.
pub fn init() -> anyhow::Result<()> {
    tracing::info!("SINGULARITY Transport v0.1.0 initializing...");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert!(super::init().is_ok());
    }

    #[test]
    fn rdma_operations_fail_loudly_without_backend() {
        // No backend is linked into this crate: operations that would
        // move bytes must error, never report a no-op success.
        let mut fabric = super::fabric::Fabric::new("mlx5_0");
        assert!(fabric.register_memory(std::ptr::null(), 4096).is_err());
        assert!(fabric.teleport_direct(0, "gpu-1", 4096).is_err());
    }
}
