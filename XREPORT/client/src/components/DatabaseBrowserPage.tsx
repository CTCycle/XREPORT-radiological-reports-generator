import { useCallback, useEffect, useMemo, useRef } from 'react';
import { RefreshCcw, Database as DatabaseIcon } from 'lucide-react';
import './DatabaseBrowserPage.css';
import { fetchBrowseConfig, fetchTableData, fetchTableList } from '../services/databaseBrowser';
import { useDatabaseBrowserState } from '../AppStateContext';

const DEFAULT_BATCH_SIZE = 200;

export default function DatabaseBrowserPage() {
    const { state, setState } = useDatabaseBrowserState();
    const sentinelRef = useRef<HTMLDivElement>(null);
    const tableScrollRef = useRef<HTMLDivElement>(null);

    const loadTableData = useCallback(
        async (tableName: string, limit: number, offset: number, append = false) => {
            if (!tableName) return;

            if (append) {
                setState(prev => ({ ...prev, loadingMore: true, error: null }));
            } else {
                setState(prev => ({
                    ...prev,
                    selectedTable: tableName,
                    limit,
                    offset: 0,
                    loading: true,
                    error: null,
                    rows: [],
                    hasMore: true
                }));
            }

            const { result, error } = await fetchTableData(tableName, limit, offset);

            if (error || !result) {
                setState(prev => ({
                    ...prev,
                    loading: false,
                    loadingMore: false,
                    error: error ?? 'Unable to load table.'
                }));
                return;
            }

            setState(prev => {
                const newRows = append ? [...prev.rows, ...result.data] : result.data;
                const newOffset = newRows.length;
                const hasMore = newRows.length < result.row_count;

                return {
                    ...prev,
                    selectedTable: tableName,
                    rows: newRows,
                    columns: result.columns,
                    rowCount: result.row_count,
                    columnCount: result.column_count,
                    displayName: result.display_name,
                    offset: newOffset,
                    loading: false,
                    loadingMore: false,
                    dataLoaded: true,
                    hasMore,
                    error: null
                };
            });
        },
        [setState],
    );

    const loadMore = useCallback(() => {
        if (state.loadingMore || state.loading || !state.hasMore || !state.selectedTable) return;
        loadTableData(state.selectedTable, state.limit, state.offset, true);
    }, [state.loadingMore, state.loading, state.hasMore, state.selectedTable, state.limit, state.offset, loadTableData]);

    // Load tables list and config on mount (but NOT data)
    useEffect(() => {
        if (state.tablesLoaded) return;
        const loadTablesAndConfig = async () => {
            // First load config
            const { config } = await fetchBrowseConfig();
            const batchSize = config?.browse_batch_size ?? DEFAULT_BATCH_SIZE;

            // Then load table list
            const { tables, error } = await fetchTableList();
            if (error) {
                setState(prev => ({ ...prev, tablesLoaded: true, limit: batchSize, error }));
                return;
            }
            const firstTable = tables.length > 0 ? tables[0].table_name : '';
            // Set tables but do NOT auto-fetch data (Rule 1: don't fetch on page load)
            setState(prev => ({
                ...prev,
                tables,
                selectedTable: firstTable,
                tablesLoaded: true,
                limit: batchSize,
                error: null
            }));
        };
        void loadTablesAndConfig();
    }, [state.tablesLoaded, setState]);

    // Intersection Observer for infinite scroll
    useEffect(() => {
        const sentinel = sentinelRef.current;
        if (!sentinel || !state.dataLoaded) return;

        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting && state.hasMore && !state.loadingMore && !state.loading) {
                    loadMore();
                }
            },
            { threshold: 0.1 }
        );

        observer.observe(sentinel);
        return () => observer.disconnect();
    }, [state.dataLoaded, state.hasMore, state.loadingMore, state.loading, loadMore]);

    const onTableChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const tableName = event.target.value;
        // Reset scroll position when changing tables
        if (tableScrollRef.current) {
            tableScrollRef.current.scrollTop = 0;
        }
        // Selecting a table always fetches data from the start
        void loadTableData(tableName, state.limit, 0);
    };

    const refresh = () => {
        if (state.selectedTable) {
            // Reset scroll position on refresh
            if (tableScrollRef.current) {
                tableScrollRef.current.scrollTop = 0;
            }
            void loadTableData(state.selectedTable, state.limit, 0);
        }
    };

    const displayedRowCount = useMemo(() => state.rows.length, [state.rows.length]);

    return (
        <div className="dbb-page">
            {/* Header */}
            <div className="dbb-header">
                <div className="dbb-title">
                    <DatabaseIcon size={22} className="dbb-title-icon" />
                    <h1>Database Browser</h1>
                </div>
                <p className="dbb-subtitle">Browse tables stored in the application database.</p>
            </div>

            {/* Controls Row */}
            <div className="dbb-controls-row">
                <div className="dbb-controls-left">
                    <div className="dbb-control">
                        <label className="dbb-label">Select Table</label>
                        <div className="dbb-select-row">
                            <select
                                className="dbb-select"
                                value={state.selectedTable}
                                onChange={onTableChange}
                                disabled={state.loading || state.tables.length === 0}
                            >
                                {state.tables.map(table => (
                                    <option key={table.table_name} value={table.table_name}>
                                        {table.display_name}
                                    </option>
                                ))}
                            </select>
                            <button
                                className="dbb-refresh-btn"
                                onClick={refresh}
                                disabled={state.loading || !state.selectedTable}
                                title="Refresh data"
                            >
                                <RefreshCcw size={16} className={state.loading ? 'spinning' : ''} />
                            </button>
                        </div>
                    </div>
                </div>

                <div className="dbb-stats-row">
                    <span className="dbb-stats-label">Statistics</span>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Rows:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? state.rowCount : '-'}</span>
                    </div>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Columns:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? state.columnCount : '-'}</span>
                    </div>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Table:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? (state.displayName || '-') : '-'}</span>
                    </div>
                    {state.dataLoaded && (
                        <div className="dbb-stat-item">
                            <span className="dbb-stat-name">Loaded:</span>
                            <span className="dbb-stat-value">{displayedRowCount} / {state.rowCount}</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {state.error && <div className="dbb-error">{state.error}</div>}

            {/* Data Table */}
            <div className="dbb-table-wrapper">
                {state.loading ? (
                    <div className="dbb-loading">Loading...</div>
                ) : !state.dataLoaded ? (
                    <div className="dbb-empty">
                        Select a table and click refresh, or choose a different table to view data.
                    </div>
                ) : state.rows.length > 0 ? (
                    <div className="dbb-table-scroll" ref={tableScrollRef}>
                        <table className="dbb-table">
                            <thead>
                                <tr>
                                    <th className="dbb-row-num-header">#</th>
                                    {state.columns.map(col => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {state.rows.map((row, idx) => (
                                    <tr key={idx}>
                                        <td className="dbb-row-num">{idx + 1}</td>
                                        {state.columns.map(col => (
                                            <td key={col}>
                                                {row[col] !== null && row[col] !== undefined ? String(row[col]) : ''}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        {/* Infinite scroll sentinel and loading indicator */}
                        <div ref={sentinelRef} className="dbb-scroll-sentinel">
                            {state.loadingMore && (
                                <div className="dbb-load-more">
                                    <div className="dbb-load-more-spinner" />
                                    <span>Loading more...</span>
                                </div>
                            )}
                            {!state.hasMore && state.rows.length > 0 && (
                                <div className="dbb-end-of-data">
                                    All {state.rowCount} rows loaded
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="dbb-empty">
                        {state.tablesLoaded && state.tables.length === 0
                            ? 'No tables found in the connected database.'
                            : 'No data to display.'}
                    </div>
                )}
            </div>
        </div>
    );
}
