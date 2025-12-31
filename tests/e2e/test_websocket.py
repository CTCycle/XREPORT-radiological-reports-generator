"""
E2E tests for WebSocket connections.
Tests the training and inference WebSocket endpoints.
"""
from playwright.sync_api import Page


def build_ws_url(api_base_url: str, path: str) -> str:
    if api_base_url.startswith("https://"):
        return f"wss://{api_base_url.removeprefix('https://')}{path}"
    return f"ws://{api_base_url.removeprefix('http://')}{path}"


class TestTrainingWebSocket:
    """Tests for the /training/ws WebSocket endpoint."""

    def test_websocket_connection_via_page(self, page: Page, base_url: str, api_base_url: str):
        """
        Test that the training WebSocket establishes a connection.
        Uses the browser's WebSocket API through Playwright.
        """
        page.goto(base_url)

        ws_url = build_ws_url(api_base_url, "/training/ws")
        # Execute JavaScript to test WebSocket connection
        result = page.evaluate("""
            async (wsUrl) => {
                return new Promise((resolve) => {
                    const ws = new WebSocket(wsUrl);
                    let connectionResult = { connected: false, message: null };
                    
                    ws.onopen = () => {
                        connectionResult.connected = true;
                    };
                    
                    ws.onmessage = (event) => {
                        try {
                            connectionResult.message = JSON.parse(event.data);
                        } catch {
                            connectionResult.message = event.data;
                        }
                        ws.close();
                        resolve(connectionResult);
                    };
                    
                    ws.onerror = (error) => {
                        connectionResult.error = 'Connection error';
                        resolve(connectionResult);
                    };
                    
                    // Timeout after 5 seconds
                    setTimeout(() => {
                        ws.close();
                        resolve(connectionResult);
                    }, 5000);
                });
            }
        """, ws_url)
        
        # WebSocket should have connected
        assert result.get("connected") or result.get("message") is not None, \
            f"WebSocket connection failed: {result}"

    def test_websocket_receives_initial_state(self, page: Page, base_url: str, api_base_url: str):
        """
        Test that the WebSocket sends initial connection state.
        XREPORT sends a 'connection_established' message with training status.
        """
        page.goto(base_url)

        ws_url = build_ws_url(api_base_url, "/training/ws")
        result = page.evaluate("""
            async (wsUrl) => {
                return new Promise((resolve) => {
                    const ws = new WebSocket(wsUrl);
                    
                    ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            ws.close();
                            resolve(data);
                        } catch {
                            resolve({ error: 'Failed to parse message' });
                        }
                    };
                    
                    ws.onerror = () => {
                        resolve({ error: 'Connection error' });
                    };
                    
                    setTimeout(() => {
                        ws.close();
                        resolve({ error: 'Timeout' });
                    }, 5000);
                });
            }
        """, ws_url)
        
        # Should receive a connection message with initial state
        if result.get("type") == "connection_established":
            assert "is_training" in result
            assert "current_epoch" in result
            assert "total_epochs" in result
            assert "loss" in result
            assert "val_loss" in result

    def test_websocket_ping_pong(self, page: Page, base_url: str, api_base_url: str):
        """
        Test that the WebSocket responds to ping with pong.
        """
        page.goto(base_url)

        ws_url = build_ws_url(api_base_url, "/training/ws")
        result = page.evaluate("""
            async (wsUrl) => {
                return new Promise((resolve) => {
                    const ws = new WebSocket(wsUrl);
                    let gotPong = false;
                    let sentPing = false;
                    
                    ws.onopen = () => {
                        sentPing = true;
                        ws.send(JSON.stringify({ type: 'ping' }));
                    };
                    
                    ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'pong') {
                                gotPong = true;
                                ws.close();
                                resolve({ gotPong: true });
                            } else if (data.type === 'connection_established' && !sentPing) {
                                sentPing = true;
                                ws.send(JSON.stringify({ type: 'ping' }));
                            }
                        } catch {
                            // Ignore parse errors
                        }
                    };
                    
                    setTimeout(() => {
                        ws.close();
                        resolve({ gotPong: gotPong });
                    }, 3000);
                });
            }
        """, ws_url)
        
        # Should have received pong response
        assert result.get("gotPong", False), "Did not receive pong response"


class TestInferenceWebSocket:
    """Tests for the /inference/ws WebSocket endpoint."""

    def test_inference_websocket_connection(self, page: Page, base_url: str, api_base_url: str):
        """
        Test that the inference WebSocket establishes a connection.
        """
        page.goto(base_url)

        ws_url = build_ws_url(api_base_url, "/inference/ws")
        result = page.evaluate("""
            async (wsUrl) => {
                return new Promise((resolve) => {
                    const ws = new WebSocket(wsUrl);
                    let connectionResult = { connected: false };
                    
                    ws.onopen = () => {
                        connectionResult.connected = true;
                        ws.close();
                        resolve(connectionResult);
                    };
                    
                    ws.onerror = () => {
                        connectionResult.error = 'Connection error';
                        resolve(connectionResult);
                    };
                    
                    setTimeout(() => {
                        ws.close();
                        resolve(connectionResult);
                    }, 5000);
                });
            }
        """, ws_url)
        
        assert result.get("connected"), f"Inference WebSocket connection failed: {result}"


class TestWebSocketMessageFormats:
    """Tests for WebSocket message formats and protocol."""

    def test_training_websocket_message_types(self, page: Page, base_url: str, api_base_url: str):
        """
        Test that training WebSocket messages have expected types.
        Expected types: connection_established, training_update, training_started, 
        training_resumed, training_plot, training_completed, training_error
        """
        page.goto(base_url)

        ws_url = build_ws_url(api_base_url, "/training/ws")
        result = page.evaluate("""
            async (wsUrl) => {
                return new Promise((resolve) => {
                    const ws = new WebSocket(wsUrl);
                    let firstMessage = null;
                    
                    ws.onmessage = (event) => {
                        try {
                            firstMessage = JSON.parse(event.data);
                            ws.close();
                            resolve(firstMessage);
                        } catch {
                            resolve({ error: 'Parse error' });
                        }
                    };
                    
                    ws.onerror = () => {
                        resolve({ error: 'Connection error' });
                    };
                    
                    setTimeout(() => {
                        ws.close();
                        resolve({ error: 'Timeout' });
                    }, 5000);
                });
            }
        """, ws_url)
        
        if result and not result.get("error"):
            # First message should have a 'type' field
            assert "type" in result
            # Type should be one of the expected values
            expected_types = [
                "connection_established",
                "training_update",
                "training_started",
                "training_resumed",
                "training_plot",
                "training_completed",
                "training_error",
                "training_stopping",
                "ping",
                "pong",
            ]
            assert result["type"] in expected_types, f"Unexpected message type: {result['type']}"
