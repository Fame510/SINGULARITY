//! SINGULARITY Transport Fabric
//!
//! High-performance RDMA/RoCE v2 transport layer for zero-copy
//! GPU-to-GPU KV-cache teleportation.
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
}
