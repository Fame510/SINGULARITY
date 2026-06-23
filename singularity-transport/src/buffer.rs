/// Shared memory buffer management for RDMA operations.
use super::page::ContextPage;

pub struct PageBuffer {
    pages: Vec<ContextPage>,
    total_size: usize,
}

impl PageBuffer {
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            total_size: 0,
        }
    }

    pub fn add_page(&mut self, page: ContextPage) {
        self.total_size += page.size;
        self.pages.push(page);
    }

    pub fn find_page(&self, page_id: u64) -> Option<&ContextPage> {
        self.pages.iter().find(|p| p.page_id == page_id)
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}
