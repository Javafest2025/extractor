# RabbitMQ Integration for ScholarAI Extractor

This document explains the RabbitMQ integration that has been implemented in the ScholarAI Extractor service, following the same pattern as the paper-search service.

## Overview

The extractor service now supports asynchronous message processing through RabbitMQ, allowing it to:
1. Receive extraction requests from the project service
2. Process PDFs asynchronously
3. Send completion results back to the project service

## Architecture

### Message Flow

```
Project Service → RabbitMQ → Extractor Service → RabbitMQ → Project Service
     (Request)     (Queue)      (Processing)      (Queue)     (Result)
```

### Queue Configuration

- **Exchange**: `scholarai.exchange` (Topic Exchange)
- **Request Queue**: `scholarai.extraction.queue`
- **Request Routing Key**: `scholarai.extraction`
- **Result Queue**: `scholarai.extraction.completed.queue`
- **Result Routing Key**: `scholarai.extraction.completed`

## Implementation Details

### 1. Configuration (`app/config.py`)

The RabbitMQ configuration is defined in the Settings class:

```python
# RabbitMQ Configuration
rabbitmq_host: str = "localhost"
rabbitmq_port: int = 5672
rabbitmq_user: Optional[str] = None
rabbitmq_password: Optional[str] = None
rabbitmq_exchange: str = "scholarai.exchange"
rabbitmq_extraction_queue: str = "scholarai.extraction.queue"
rabbitmq_extraction_completed_queue: str = "scholarai.extraction.completed.queue"
rabbitmq_extraction_routing_key: str = "scholarai.extraction"
rabbitmq_extraction_completed_routing_key: str = "scholarai.extraction.completed"
```

### 2. Connection Management (`app/services/messaging/connection.py`)

The `RabbitMQConnection` class handles:
- Connection establishment and management
- Queue and exchange setup
- Message publishing
- Connection cleanup

### 3. Message Consumer (`app/services/messaging/consumer.py`)

The `ScholarAIConsumer` class:
- Manages the message consumption lifecycle
- Routes messages to appropriate handlers
- Handles connection and queue setup

### 4. Message Handlers (`app/services/messaging/handlers.py`)

The `ExtractionMessageHandler` class:
- Validates incoming extraction messages
- Processes extraction requests using the enhanced extraction handler
- Publishes results back to the completion queue

### 5. Enhanced Extraction Handler (`app/services/extraction_handler.py`)

The `EnhancedExtractionHandler` class has been updated to:
- Store processing context in `processing_history`
- Provide `get_extraction_result()` method for result retrieval
- Handle asynchronous processing with proper result storage

## Message Format

### Request Message (from Project Service)

```json
{
  "jobId": "uuid-string",
  "paperId": "paper-uuid",
  "correlationId": "correlation-uuid",
  "b2Url": "https://b2.example.com/path/to/paper.pdf",
  "extractText": true,
  "extractFigures": true,
  "extractTables": true,
  "extractEquations": true,
  "extractCode": true,
  "extractReferences": true,
  "useOcr": true,
  "detectEntities": true
}
```

### Response Message (to Project Service)

```json
{
  "jobId": "uuid-string",
  "paperId": "paper-uuid",
  "correlationId": "correlation-uuid",
  "status": "COMPLETED",
  "completedAt": "2024-01-01T12:00:00Z",
  "extractionCoverage": 0.85,
  "extractionResult": {
    // Full extraction result object
  },
  "message": "Extraction completed successfully"
}
```

## Usage

### Starting the Consumer

The RabbitMQ consumer can be started in several ways:

1. **As part of the FastAPI application** (recommended):
   ```python
   # The consumer is automatically started when the FastAPI app starts
   # if RabbitMQ credentials are configured
   ```

2. **Standalone consumer**:
   ```bash
   cd AI-Agents/extractor
   python -m app.services.rabbitmq_consumer
   ```

3. **Background worker**:
   ```python
   from app.services.background_worker import background_worker
   background_worker.start()
   ```

### Testing the Integration

Run the test script to verify the integration:

```bash
cd AI-Agents/extractor
python test_rabbitmq_integration.py
```

## Environment Configuration

Set the following environment variables in your `.env` file:

```env
# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=your_username
RABBITMQ_PASSWORD=your_password
```

## Dependencies

The following dependencies are required:

```txt
aio-pika==9.3.1  # Async RabbitMQ client
```

## Error Handling

The integration includes comprehensive error handling:

1. **Connection failures**: Automatic retry with exponential backoff
2. **Message validation**: Invalid messages are rejected without requeuing
3. **Processing errors**: Failed extractions are logged and reported
4. **Network issues**: Transient errors trigger message requeuing

## Monitoring

The integration provides several monitoring endpoints:

- `/api/v1/queue-status`: Check RabbitMQ connection status
- Performance metrics in the extraction handler
- Detailed logging for debugging

## Comparison with Paper-Search Service

The extractor service follows the same architectural patterns as the paper-search service:

| Aspect | Paper-Search | Extractor |
|--------|-------------|-----------|
| Exchange | `scholarai.exchange` | `scholarai.exchange` |
| Request Queue | `scholarai.websearch.queue` | `scholarai.extraction.queue` |
| Result Queue | `scholarai.websearch.completed.queue` | `scholarai.extraction.completed.queue` |
| Message Handler | `WebSearchMessageHandler` | `ExtractionMessageHandler` |
| Processing | `WebSearchAgent` | `EnhancedExtractionHandler` |

## Troubleshooting

### Common Issues

1. **Connection refused**: Check if RabbitMQ server is running
2. **Authentication failed**: Verify username/password in environment variables
3. **Queue not found**: Ensure the project service has created the required queues
4. **Message format errors**: Check that the message format matches the expected schema

### Debugging

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

### Manual Testing

You can manually test the integration using the test script:
```bash
python test_rabbitmq_integration.py
```

## Future Enhancements

Potential improvements for the RabbitMQ integration:

1. **Message persistence**: Ensure messages survive RabbitMQ restarts
2. **Dead letter queues**: Handle failed messages gracefully
3. **Message prioritization**: Prioritize urgent extraction requests
4. **Load balancing**: Distribute processing across multiple extractor instances
5. **Health checks**: Monitor RabbitMQ connection health
6. **Metrics collection**: Track message processing performance

## Conclusion

The RabbitMQ integration provides a robust, scalable solution for asynchronous PDF extraction processing. It follows established patterns from the paper-search service and integrates seamlessly with the existing extraction pipeline.
