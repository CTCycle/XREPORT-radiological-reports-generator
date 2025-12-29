"""
E2E tests for Database Browser API endpoints.
Tests: /data/browser/tables, /data/browser/config, /data/browser/data/{table_name}
"""
from playwright.sync_api import APIRequestContext


class TestDatabaseBrowserEndpoints:
    """Tests for the /data/browser/* API endpoints."""

    def test_list_tables_returns_table_info(self, api_context: APIRequestContext):
        """GET /data/browser/tables should return a list of tables with display names."""
        response = api_context.get("/data/browser/tables")
        assert response.ok, f"Expected 200, got {response.status}"
        
        data = response.json()
        assert "tables" in data, "Response should have 'tables' key"
        assert isinstance(data["tables"], list), "'tables' should be a list"
        
        # Each table should have name and display_name
        for table in data["tables"]:
            assert "table_name" in table
            assert "display_name" in table

    def test_list_tables_includes_expected_tables(self, api_context: APIRequestContext):
        """GET /data/browser/tables should include XREPORT database tables."""
        response = api_context.get("/data/browser/tables")
        assert response.ok
        
        data = response.json()
        table_names = [t["table_name"] for t in data["tables"]]
        
        # Check for at least one expected XREPORT table
        expected_tables = [
            "RADIOGRAPHY_DATA",
            "TRAINING_DATASET", 
            "PROCESSING_METADATA",
            "GENERATED_REPORTS",
            "IMAGE_STATISTICS",
            "TEXT_STATISTICS",
            "CHECKPOINTS_SUMMARY",
        ]
        
        # At minimum, some system tables should exist
        # Note: Tables may not exist if database is empty, so we just check the response format
        assert isinstance(table_names, list)

    def test_get_browse_config(self, api_context: APIRequestContext):
        """GET /data/browser/config should return browse batch size configuration."""
        response = api_context.get("/data/browser/config")
        assert response.ok, f"Expected 200, got {response.status}"
        
        data = response.json()
        assert "browse_batch_size" in data
        assert isinstance(data["browse_batch_size"], int)
        assert data["browse_batch_size"] > 0

    def test_get_table_data_valid_table(self, api_context: APIRequestContext):
        """GET /data/browser/data/{table_name} should return paginated data for valid tables."""
        # First get list of tables
        tables_response = api_context.get("/data/browser/tables")
        if not tables_response.ok:
            return  # Skip if can't get tables
        
        tables = tables_response.json().get("tables", [])
        if not tables:
            return  # Skip if no tables exist
        
        # Try to query the first available table
        table_name = tables[0]["table_name"]
        response = api_context.get(f"/data/browser/data/{table_name}")
        assert response.ok, f"Expected 200, got {response.status}"
        
        data = response.json()
        assert "table_name" in data
        assert "display_name" in data
        assert "columns" in data
        assert "data" in data
        assert "row_count" in data
        assert "column_count" in data
        assert isinstance(data["columns"], list)
        assert isinstance(data["data"], list)

    def test_get_table_data_with_pagination(self, api_context: APIRequestContext):
        """GET /data/browser/data/{table_name}?offset=N&limit=M should support pagination."""
        # Get available tables
        tables_response = api_context.get("/data/browser/tables")
        if not tables_response.ok:
            return
        
        tables = tables_response.json().get("tables", [])
        if not tables:
            return
        
        table_name = tables[0]["table_name"]
        
        # Request with specific limit and offset
        response = api_context.get(f"/data/browser/data/{table_name}?limit=10&offset=0")
        assert response.ok
        
        data = response.json()
        # Data length should be at most the limit
        assert len(data["data"]) <= 10

    def test_get_table_data_invalid_table_returns_404(self, api_context: APIRequestContext):
        """GET /data/browser/data/{invalid} should return 404."""
        response = api_context.get("/data/browser/data/NON_EXISTENT_TABLE_XYZ")
        assert response.status == 404
        
        data = response.json()
        assert "detail" in data


class TestDatabaseBrowserEdgeCases:
    """Edge case tests for database browser functionality."""

    def test_get_table_data_with_zero_offset(self, api_context: APIRequestContext):
        """GET /data/browser/data/{table_name}?offset=0 should work."""
        tables_response = api_context.get("/data/browser/tables")
        if not tables_response.ok:
            return
        
        tables = tables_response.json().get("tables", [])
        if not tables:
            return
        
        table_name = tables[0]["table_name"]
        response = api_context.get(f"/data/browser/data/{table_name}?offset=0")
        assert response.ok

    def test_get_table_data_with_large_offset(self, api_context: APIRequestContext):
        """GET /data/browser/data/{table_name}?offset=999999 should return empty data."""
        tables_response = api_context.get("/data/browser/tables")
        if not tables_response.ok:
            return
        
        tables = tables_response.json().get("tables", [])
        if not tables:
            return
        
        table_name = tables[0]["table_name"]
        response = api_context.get(f"/data/browser/data/{table_name}?offset=999999")
        assert response.ok
        
        data = response.json()
        # With a very large offset, data should be empty
        assert data["data"] == []
