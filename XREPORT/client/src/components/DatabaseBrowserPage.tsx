import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCcw, Database as DatabaseIcon } from 'lucide-react';
import './DatabaseBrowserPage.css';
import { fetchTableData, fetchTableList, TableInfo } from '../services/databaseBrowser';

interface BrowserState {
    tables: TableInfo[];
    selectedTable: string;
    rows: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    columnCount: number;
    displayName: string;
    limit: number;
    offset: number;
    loading: boolean;
    error: string | null;
    tablesLoaded: boolean;
}

const PAGE_SIZES = [50, 200, 500, 1000] as const;

export default function DatabaseBrowserPage() {
    const [state, setState] = useState<BrowserState>({
        tables: [],
        selectedTable: '',
        rows: [],
        columns: [],
        rowCount: 0,
        columnCount: 0,
        displayName: '',
        limit: 200,
        offset: 0,
        loading: false,
        error: null,
        tablesLoaded: false,
    });

    const pageIndex = useMemo(() => Math.floor(state.offset / state.limit), [state.offset, state.limit]);
    const pageStart = useMemo(() => (state.rowCount === 0 ? 0 : state.offset + 1), [state.offset, state.rowCount]);
    const pageEnd = useMemo(
        () => Math.min(state.offset + state.rows.length, state.rowCount),
        [state.offset, state.rows.length, state.rowCount],
    );

    const canGoPrevious = useMemo(() => state.offset > 0 && !state.loading, [state.offset, state.loading]);
    const canGoNext = useMemo(
        () => state.offset + state.limit < state.rowCount && !state.loading,
        [state.offset, state.limit, state.rowCount, state.loading],
    );

    const loadTableData = useCallback(
        async (tableName: string, limit: number, offset: number) => {
            if (!tableName) return;
            setState(prev => ({ ...prev, selectedTable: tableName, limit, offset, loading: true, error: null }));
            const { result, error } = await fetchTableData(tableName, limit, offset);
            if (error || !result) {
                setState(prev => ({
                    ...prev,
                    selectedTable: tableName,
                    rows: [],
                    columns: [],
                    rowCount: 0,
                    columnCount: 0,
                    displayName: '',
                    loading: false,
                    error: error ?? 'Unable to load table.',
                }));
                return;
            }
            setState(prev => ({
                ...prev,
                selectedTable: tableName,
                rows: result.data,
                columns: result.columns,
                rowCount: result.row_count,
                columnCount: result.column_count,
                displayName: result.display_name,
                loading: false,
                error: null,
            }));
        },
        [],
    );

    useEffect(() => {
        if (state.tablesLoaded) return;
        const loadTables = async () => {
            const { tables, error } = await fetchTableList();
            if (error) {
                setState(prev => ({ ...prev, tablesLoaded: true, error }));
                return;
            }
            const firstTable = tables.length > 0 ? tables[0].table_name : '';
            setState(prev => ({ ...prev, tables, selectedTable: firstTable, tablesLoaded: true, error: null }));
            if (firstTable) {
                await loadTableData(firstTable, state.limit, 0);
            }
        };
        void loadTables();
    }, [state.tablesLoaded, state.limit, loadTableData]);

    const onTableChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const tableName = event.target.value;
        void loadTableData(tableName, state.limit, 0);
    };

    const onPageSizeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const nextLimit = Number(event.target.value);
        void loadTableData(state.selectedTable, nextLimit, 0);
    };

    const refresh = () => {
        void loadTableData(state.selectedTable, state.limit, state.offset);
    };

    const goPrevious = () => {
        const nextOffset = Math.max(0, state.offset - state.limit);
        void loadTableData(state.selectedTable, state.limit, nextOffset);
    };

    const goNext = () => {
        const nextOffset = state.offset + state.limit;
        void loadTableData(state.selectedTable, state.limit, nextOffset);
    };

    return (
        <div className="dbb-page">
            <div className="dbb-header">
                <div className="dbb-title">
                    <div className="dbb-title-row">
                        <DatabaseIcon size={18} />
                        <h1>Database Browser</h1>
                    </div>
                    <p>Browse tables stored in the application database.</p>
                </div>
            </div>

            <div className="dbb-controls">
                <div className="dbb-control">
                    <label className="dbb-label">Table</label>
                    <div className="dbb-row">
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
                        <button className="dbb-icon-btn" onClick={refresh} disabled={state.loading || !state.selectedTable}>
                            <RefreshCcw size={16} />
                        </button>
                    </div>
                </div>

                <div className="dbb-control">
                    <label className="dbb-label">Page Size</label>
                    <select className="dbb-select" value={state.limit} onChange={onPageSizeChange} disabled={state.loading}>
                        {PAGE_SIZES.map(size => (
                            <option key={size} value={size}>
                                {size}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="dbb-stats">
                    <div className="dbb-stat">
                        <span className="dbb-stat-label">Rows</span>
                        <span className="dbb-stat-value">{state.rowCount}</span>
                    </div>
                    <div className="dbb-stat">
                        <span className="dbb-stat-label">Columns</span>
                        <span className="dbb-stat-value">{state.columnCount}</span>
                    </div>
                    <div className="dbb-stat">
                        <span className="dbb-stat-label">Table</span>
                        <span className="dbb-stat-value">{state.displayName || '-'}</span>
                    </div>
                </div>
            </div>

            <div className="dbb-pagination">
                <button className="dbb-btn" onClick={goPrevious} disabled={!canGoPrevious}>
                    Previous
                </button>
                <div className="dbb-page-info">
                    Page {state.rowCount === 0 ? 0 : pageIndex + 1} · Showing {pageStart}-{pageEnd} of {state.rowCount}
                </div>
                <button className="dbb-btn" onClick={goNext} disabled={!canGoNext}>
                    Next
                </button>
            </div>

            {state.error && <div className="dbb-error">{state.error}</div>}

            <div className="dbb-table-container">
                {state.loading ? (
                    <div className="dbb-loading">Loading…</div>
                ) : state.rows.length > 0 ? (
                    <div className="dbb-table-scroll">
                        <table className="dbb-table">
                            <thead>
                                <tr>
                                    {state.columns.map(col => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {state.rows.map((row, idx) => (
                                    <tr key={idx}>
                                        {state.columns.map(col => (
                                            <td key={col}>
                                                {row[col] !== null && row[col] !== undefined ? String(row[col]) : ''}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
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

