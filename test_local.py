#!/usr/bin/env python3
"""
Local testing script for Marketing Campaign Lambda
"""

import json
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your lambda function
from lambda_function import lambda_handler

def test_marketing_campaign():
    """Test marketing campaign generation"""
    
    # Sample marketing campaign request
    test_payload = {
        "httpMethod": "POST",
        "path": "/marketing-campaign",
        "body": json.dumps({
            "user_id": "test_marketing_user",
            "message": "Generate personalized B2B marketing campaign messages for my customer list",
            "agent_type": "marketing_campaign_agent",
            "context": """Customer Data:
1. Barton Townley at Osmii (Software Development Recruiter) - barton@osmii.com
2. Sarah Johnson at TechCorp Industries (Chief Technology Officer) - sarah@techcorp.com
3. Mike Chen at Healthcare Solutions Inc (Operations Manager)

Offering Details:
- Name: AI-Powered Business Intelligence Platform
- Objective: Streamline decision-making with real-time data insights
- Benefits: Automated reporting, predictive analytics, custom dashboards
- Target Industry: Technology
- Price Range: $10,000 - $50,000
- Duration: 3-6 months implementation"""
        })
    }
    
    print("🧪 Testing Marketing Campaign Generation...")
    print(f"📅 Test started at: {datetime.now()}")
    print("="*60)
    
    try:
        # Call lambda handler
        response = lambda_handler(test_payload, None)
        
        print(f"📊 Response Status: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            print(f"✅ Success: {body.get('success')}")
            print(f"📝 Status: {body.get('status')}")
            
            if body.get('status') == 'processing':
                print(f"🔄 Request Hash: {body.get('request_hash', '')[:12]}...")
                print(f"📈 Progress: {body.get('progress', 0)}%")
                print(f"⏱️ Estimated Duration: {body.get('estimated_duration', 0)}s")
                print(f"🔗 Poll URL: {body.get('poll_url', 'N/A')}")
                
                # Test polling
                print("\n🔍 Testing Polling...")
                test_polling(body.get('request_hash'))
                
            elif body.get('status') == 'completed':
                print(f"📄 Response Preview: {body.get('response', '')[:200]}...")
                if body.get('image_urls'):
                    print(f"🖼️ Images Generated: {len(body['image_urls'])}")
                if body.get('files_created'):
                    print(f"📁 Files Created: {len(body['files_created'])}")
        else:
            print(f"❌ Error Response: {response['body']}")
            
    except Exception as e:
        print(f"💥 Test Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)
    print(f"🏁 Test completed at: {datetime.now()}")

def test_polling(request_hash):
    """Test polling functionality"""
    if not request_hash:
        return
        
    poll_payload = {
        "httpMethod": "GET",
        "path": f"/poll/{request_hash}"
    }
    
    try:
        response = lambda_handler(poll_payload, None)
        
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            print(f"📊 Poll Status: {body.get('status')}")
            print(f"📈 Progress: {body.get('progress', 0)}%")
            print(f"💬 Message: {body.get('message', 'No message')}")
            
            if body.get('status') == 'completed':
                print(f"📄 Final Response: {body.get('response', '')[:200]}...")
        else:
            print(f"❌ Poll Error: {response['body']}")
            
    except Exception as e:
        print(f"💥 Polling Exception: {e}")

def test_simple_chat():
    """Test simple chat functionality"""
    
    test_payload = {
        "httpMethod": "POST",
        "path": "/chat", 
        "body": json.dumps({
            "user_id": "test_user",
            "message": "Hello! Can you help me with a simple marketing question?",
            "agent_type": "ai_chat_agent"
        })
    }
    
    print("🧪 Testing Simple Chat...")
    
    try:
        response = lambda_handler(test_payload, None)
        print(f"📊 Response Status: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            print(f"✅ Success: {body.get('success')}")
            print(f"📝 Response: {body.get('response', '')[:200]}...")
        
    except Exception as e:
        print(f"💥 Simple Chat Exception: {e}")

def run_all_tests():
    """Run all test scenarios"""
    print("🚀 Starting Marketing Campaign Lambda Tests")
    print("="*80)
    
    # Test 1: Simple chat
    test_simple_chat()
    print("\n" + "-"*60 + "\n")
    
    # Test 2: Marketing campaign
    test_marketing_campaign()
    print("\n" + "-"*60 + "\n")
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--marketing-only":
        test_marketing_campaign()
    elif len(sys.argv) > 1 and sys.argv[1] == "--simple-only":
        test_simple_chat()
    else:
        run_all_tests()