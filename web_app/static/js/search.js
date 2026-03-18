/**
 * Search functionality for the landing page.
 *
 * Handles search form submission and example query clicks.
 */

document.addEventListener('DOMContentLoaded', function() {
    initSearch();
});

/**
 * Initialize search functionality.
 */
function initSearch() {
    const searchForm = document.querySelector('.search-form');
    const searchInput = document.getElementById('search-input');
    const exampleList = document.getElementById('example-list');

    // Handle form submission
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const query = searchInput.value.trim();
            if (!query) {
                e.preventDefault();
                searchInput.focus();
            }
        });
    }

    // Handle example query clicks
    if (exampleList) {
        exampleList.querySelectorAll('li').forEach(function(item) {
            item.addEventListener('click', function() {
                const query = this.getAttribute('data-query');
                if (searchInput && query) {
                    searchInput.value = query;
                    searchInput.focus();
                    // Optionally auto-submit
                    // searchForm.submit();
                }
            });
        });
    }

    // Handle Enter key
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && this.value.trim()) {
                searchForm.submit();
            }
        });

        // Focus on page load
        searchInput.focus();
    }
}

/**
 * Submit a search query programmatically.
 *
 * @param {string} query - The query text.
 */
function submitSearch(query) {
    const searchInput = document.getElementById('search-input');
    const searchForm = document.querySelector('.search-form');

    if (searchInput && searchForm && query) {
        searchInput.value = query;
        searchForm.submit();
    }
}
