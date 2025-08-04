import json
import sys
import os

# Mock the AWS/Strands environment for local testing
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Import your lambda function
try:
    from lambda_function import lambda_handler
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating minimal test instead...")
    
    def lambda_handler(event, context):
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Local test - imports not available'})
        }

# Test payload based on your strands_agent_communication.py
def test_simple_chat():
    test_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "user_id": "test_user_123",
            "message": "Hello! Can you help me with marketing campaigns?",
            "agent_type": "ai_chat_agent"
        })
    }
    
    print("🧪 Testing simple chat...")
    result = lambda_handler(test_event, None)
    print(f"Status: {result['statusCode']}")
    print(f"Response: {result['body']}")

def test_marketing_campaign():
    test_event = {
        "httpMethod": "POST", 
        "body": json.dumps({
            "user_id": "marketing_test_001",
            "message": "Generate personalized marketing campaign messages",
            "agent_type": "marketing_campaign_agent",
            "context": """Customer Data:
1. John Doe at TechCorp (CEO)
2. Jane Smith at HealthInc (Director)

Offering Details:
- AI Platform
- $25k investment
- Technology industry"""
        })
    }
    
    print("\n🎯 Testing marketing campaign...")
    result = lambda_handler(test_event, None)
    print(f"Status: {result['statusCode']}")
    print(f"Response: {result['body']}")

if __name__ == "__main__":
    test_simple_chat()
    test_marketing_campaign()