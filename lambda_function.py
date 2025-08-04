import sys
sys.path.insert(0, "/mnt/efs/envs/strands_lambda/lib/python3.11/site-packages/")

import boto3
import json
import uuid
import time
import hashlib
import base64
import io
import os
from botocore.exceptions import ClientError
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from datetime import datetime, timedelta
from threading import Thread
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ADD THESE IMPORTS FOR MARKETING CAMPAIGNS
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Set EFS path for writable operations
EFS_PATH = "/mnt/efs/tmp"
os.makedirs(EFS_PATH, exist_ok=True)
os.environ["TMPDIR"] = EFS_PATH

# Import Strands built-in tools selectively to avoid file system issues
try:
    from strands_tools import use_aws
    print("✅ use_aws imported successfully")
except Exception as e:
    print(f"⚠️ Could not import use_aws: {e}")
    use_aws = None

try:
    from strands_tools import speak
    print("✅ speak imported successfully")
except Exception as e:
    print(f"⚠️ Could not import speak: {e}")
    speak = None

try:
    from strands_tools import generate_image
    print("✅ generate_image imported successfully")
except Exception as e:
    print(f"⚠️ Could not import generate_image: {e}")
    generate_image = None

try:
    from strands_tools import nova_reels
    print("✅ nova_reels imported successfully")
except Exception as e:
    print(f"⚠️ Could not import nova_reels: {e}")
    nova_reels = None

try:
    from strands_tools import image_reader
    print("✅ image_reader imported successfully")
except Exception as e:
    print(f"⚠️ Could not import image_reader: {e}")
    image_reader = None

try:
    from strands_tools import retrieve
    print("✅ retrieve imported successfully")
except Exception as e:
    print(f"⚠️ Could not import retrieve: {e}")
    retrieve = None

try:
    from strands_tools import memory
    print("✅ memory imported successfully")
except Exception as e:
    print(f"⚠️ Could not import memory: {e}")
    memory = None

try:
    from strands_tools import file_read
    print("✅ file_read imported successfully")
except Exception as e:
    print(f"⚠️ Could not import file_read: {e}")
    file_read = None

try:
    from strands_tools import file_write
    print("✅ file_write imported successfully")
except Exception as e:
    print(f"⚠️ Could not import file_write: {e}")
    file_write = None

try:
    from strands_tools import http_request
    print("✅ http_request imported successfully")
except Exception as e:
    print(f"⚠️ Could not import http_request: {e}")
    http_request = None

try:
    from strands_tools import python_repl
    print("✅ python_repl imported successfully")
except Exception as e:
    print(f"⚠️ Could not import python_repl: {e}")
    python_repl = None

try:
    from strands_tools import calculator
    print("✅ calculator imported successfully")
except Exception as e:
    print(f"⚠️ Could not import calculator: {e}")
    calculator = None

try:
    from strands_tools import current_time
    print("✅ current_time imported successfully")
except Exception as e:
    print(f"⚠️ Could not import current_time: {e}")
    current_time = None

try:
    from strands_tools import workflow
    print("✅ workflow imported successfully")
except Exception as e:
    print(f"⚠️ Could not import workflow: {e}")
    workflow = None

try:
    from strands_tools import use_llm
    print("✅ use_llm imported successfully")
except Exception as e:
    print(f"⚠️ Could not import use_llm: {e}")
    use_llm = None

# Set environment variable to bypass tool consent for automation
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ========== ADD MARKETING CAMPAIGN CLASSES HERE ==========

class Environment(Enum):
    """Deployment environments for marketing campaigns"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class MessageComplexity(Enum):
    """Message complexity levels for campaign generation"""
    SIMPLE = "simple"        # Basic customer data
    MODERATE = "moderate"    # Some personalization
    COMPLEX = "complex"      # Heavy personalization with enriched data

@dataclass
class MarketingModelConfig:
    model_id: str
    environment: str
    cost_per_1k_tokens: float
    description: str

class MarketingCampaignManager:
    """Integrated marketing campaign management for the existing Lambda system"""
    
    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        self.models = {
            Environment.DEVELOPMENT: MarketingModelConfig(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                environment="development",
                cost_per_1k_tokens=0.003,
                description="Claude 3.7 Sonnet - Development campaigns"
            ),
            Environment.PRODUCTION: MarketingModelConfig(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Matches existing system
                environment="production", 
                cost_per_1k_tokens=0.015,
                description="Claude 4 Sonnet - Production campaigns"
            )
        }
        
        self.current_model = self.models[environment]
        print(f"🎯 Marketing Campaign Manager initialized: {self.current_model.description}")
    
    def get_marketing_campaign_prompt(self, customer_data: dict, offering_details: dict, 
                                    message_type: str = "quick_connect") -> str:
        """Generate marketing campaign prompt that integrates with existing agent system"""
        
        # Detect customer attributes
        industry = self._detect_industry(customer_data)
        designation_level = self._detect_designation_level(customer_data)
        
        # Customer context
        customer_context = f"Customer: {customer_data.get('first_name', 'there')}"
        if customer_data.get('company'):
            customer_context += f" at {customer_data['company']}"
        if customer_data.get('position'):
            customer_context += f" ({customer_data['position']})"
        
        if message_type == "quick_connect":
            return self._generate_quick_connect_prompt(
                customer_context, industry, designation_level, offering_details
            )
        else:
            return self._generate_value_proposition_prompt(
                customer_context, industry, designation_level, offering_details
            )
    
    def _generate_quick_connect_prompt(self, customer_context: str, industry: str, 
                                     designation_level: str, offering_details: dict) -> str:
        """Generate quick connect prompt (50-75 words)"""
        
        pain_points_by_industry = {
            'technology': "scaling infrastructure while managing costs and staying ahead of AI revolution",
            'healthcare': "improving patient outcomes while reducing operational costs and ensuring compliance", 
            'finance': "maximizing ROI while minimizing risk in uncertain market conditions",
            'retail': "boosting customer lifetime value while dominating competitive market share",
            'manufacturing': "reducing production costs while improving quality standards and safety",
            'education': "improving student outcomes while optimizing budget allocation and resources",
            'consulting': "expanding client base while increasing project profitability and reputation",
            'other': "improving business operations while maintaining competitive advantage"
        }
        
        industry_pain = pain_points_by_industry.get(industry, pain_points_by_industry['other'])
        
        base_prompt = """
You are a world-class B2B cold messaging specialist. Create a personalized, professional, and truthful outreach message.

🚨 CRITICAL RULES:
- NEVER invent statistics or fake achievements
- NEVER claim "we helped Company X do Y" without proof  
- Use phrases like "designed to help" instead of definitive claims
- Focus on asking questions, not making bold statements
- Be genuinely curious about their challenges

🎯 TASK: Generate a 50-75 word quick connect message that sparks curiosity.
"""
        
        return f"""{base_prompt}

👤 CUSTOMER: {customer_context}
Industry: {industry.title()}
Seniority: {designation_level.replace('_', ' ').title()}

💼 YOUR OFFERING:
Name: {offering_details.get('name', 'Our solution')}
Objective: {offering_details.get('objective', 'Improve business operations')}
Benefits: {offering_details.get('benefits', 'Enhanced efficiency and performance')}

🎯 INDUSTRY CONTEXT:
{industry.title()} professionals often struggle with {industry_pain}.

📋 REQUIREMENTS:
- Exactly 50-75 words
- Include ONE thoughtful question about their challenges
- Professional tone for {designation_level.replace('_', ' ')} level
- NO statistics or unverifiable claims
- Show understanding of {industry} industry challenges

Generate the personalized message now:"""

    def _generate_value_proposition_prompt(self, customer_context: str, industry: str,
                                         designation_level: str, offering_details: dict) -> str:
        """Generate value proposition prompt (175-200 words)"""
        
        base_prompt = """
You are a world-class B2B cold messaging specialist. Create a detailed, consultative outreach message.

🚨 CRITICAL RULES:
- NEVER invent statistics or fake achievements
- NEVER claim "we helped Company X do Y" without proof
- Use phrases like "designed to help" and "potential to improve"
- Ask 2-3 thoughtful questions about their situation
- Position as peer consultant, not vendor

🎯 TASK: Generate a 175-200 word value proposition message.
"""
        
        return f"""{base_prompt}

👤 CUSTOMER: {customer_context}
Industry: {industry.title()}
Seniority: {designation_level.replace('_', ' ').title()}

💼 YOUR OFFERING:
Name: {offering_details.get('name', 'Our solution')}
Primary Objective: {offering_details.get('objective', 'Improve business operations')}
Key Benefits: {offering_details.get('benefits', 'Enhanced efficiency and performance')}
Investment Range: {offering_details.get('price_range', 'Flexible pricing')}
Timeline: {offering_details.get('duration', 'Customizable implementation')}

📋 REQUIREMENTS:
- Exactly 175-200 words
- 2-3 thoughtful questions about their situation
- Industry-specific insights for {industry}
- Appropriate tone for {designation_level.replace('_', ' ')} level
- NO false claims or statistics
- Consultative, peer-to-peer approach

Generate the personalized message now:"""

    def _detect_industry(self, customer_data: dict) -> str:
        """Detect customer industry from company and position"""
        company = customer_data.get('company', '').lower()
        position = customer_data.get('position', '').lower()
        
        industry_keywords = {
            'technology': ['tech', 'software', 'IT', 'development', 'engineering', 'digital', 'ai', 'data', 'cloud'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic', 'pharma', 'care', 'biotech'],
            'finance': ['bank', 'finance', 'investment', 'insurance', 'accounting', 'fintech', 'trading'],
            'retail': ['retail', 'sales', 'store', 'commerce', 'merchandising', 'ecommerce'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'automotive'],
            'education': ['education', 'school', 'university', 'academic', 'learning', 'training'],
            'consulting': ['consulting', 'consultant', 'advisory', 'strategy', 'management']
        }
        
        text = f"{company} {position}"
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                return industry
        
        return 'other'
    
    def _detect_designation_level(self, customer_data: dict) -> str:
        """Detect seniority level from position"""
        position = customer_data.get('position', '').lower()
        
        if any(title in position for title in ['ceo', 'cto', 'cfo', 'cmo', 'president', 'founder', 'chief']):
            return 'c_level'
        elif any(title in position for title in ['director', 'head of', 'vp', 'vice president']):
            return 'director' 
        elif any(title in position for title in ['manager', 'lead', 'supervisor']):
            return 'manager'
        elif any(title in position for title in ['senior', 'specialist', 'analyst', 'engineer']):
            return 'specialist'
        else:
            return 'junior'

    def process_customer_batch(self, customers: List[dict], offering_details: dict, 
                             message_type: str = "both") -> List[dict]:
        """Process batch of customers for campaign generation"""
        results = []
        
        for customer in customers:
            try:
                customer_result = {
                    'customer': customer,
                    'industry': self._detect_industry(customer),
                    'designation_level': self._detect_designation_level(customer),
                    'messages': {}
                }
                
                if message_type in ["quick_connect", "both"]:
                    customer_result['messages']['quick_connect'] = self.get_marketing_campaign_prompt(
                        customer, offering_details, "quick_connect"
                    )
                
                if message_type in ["value_proposition", "both"]:
                    customer_result['messages']['value_proposition'] = self.get_marketing_campaign_prompt(
                        customer, offering_details, "value_proposition"
                    )
                
                results.append(customer_result)
                
            except Exception as e:
                results.append({
                    'customer': customer,
                    'error': str(e),
                    'messages': {}
                })
        
        return results

# ========== END MARKETING CAMPAIGN CLASSES ==========

class EnhancedCacheManager:
    """Enhanced cache manager with comprehensive request handling"""
    
    def __init__(self, dynamodb_resource):
        self.dynamodb = dynamodb_resource
        self.cache_table_name = 'streaming-ai-request-cache'
        self.response_table_name = 'streaming-ai-responses'
        self.progress_table_name = 'streaming-ai-progress'
        self._ensure_cache_tables_exist()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.active_requests = {}  # Track active processing requests
        
    def _ensure_cache_tables_exist(self):
        """Ensure all cache tables exist with proper schemas"""
        
        # Main cache table - stores request metadata and final responses
        try:
            self.cache_table = self.dynamodb.Table(self.cache_table_name)
            self.cache_table.load()
            print(f"✅ Using existing cache table: {self.cache_table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"🔧 Creating cache table: {self.cache_table_name}")
                self.cache_table = self.dynamodb.create_table(
                    TableName=self.cache_table_name,
                    KeySchema=[
                        {'AttributeName': 'request_hash', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'request_hash', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                self.cache_table.wait_until_exists()
                print("✅ Cache table created successfully")
            else:
                self.cache_table = None

        # Responses table - stores full response data with TTL
        try:
            self.response_table = self.dynamodb.Table(self.response_table_name)
            self.response_table.load()
            print(f"✅ Using existing response table: {self.response_table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"🔧 Creating response table: {self.response_table_name}")
                self.response_table = self.dynamodb.create_table(
                    TableName=self.response_table_name,
                    KeySchema=[
                        {'AttributeName': 'request_hash', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'request_hash', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                self.response_table.wait_until_exists()
                print("✅ Response table created successfully")
            else:
                self.response_table = None

        # Progress tracking table - for real-time progress updates
        try:
            self.progress_table = self.dynamodb.Table(self.progress_table_name)
            self.progress_table.load()
            print(f"✅ Using existing progress table: {self.progress_table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"🔧 Creating progress table: {self.progress_table_name}")
                self.progress_table = self.dynamodb.create_table(
                    TableName=self.progress_table_name,
                    KeySchema=[
                        {'AttributeName': 'request_hash', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'request_hash', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                self.progress_table.wait_until_exists()
                print("✅ Progress table created successfully")
            else:
                self.progress_table = None

    def _generate_request_hash(self, payload):
        """Generate deterministic hash for request deduplication"""
        # Create normalized payload for hashing
        normalized_payload = {
            'user_id': payload.get('user_id', ''),
            'message': payload.get('message', '').strip(),
            'agent_type': payload.get('agent_type', ''),
            'context': payload.get('context', '').strip(),
            'file_upload_info': self._normalize_file_upload(payload.get('file_upload'))
        }
        
        # Create deterministic string
        payload_string = json.dumps(normalized_payload, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(payload_string.encode('utf-8')).hexdigest()

    def _normalize_file_upload(self, file_upload):
        """Normalize file upload info for hashing (exclude content, include metadata)"""
        if not file_upload:
            return None
            
        if isinstance(file_upload, dict):
            return {
                'filename': file_upload.get('filename', ''),
                'content_type': file_upload.get('contentType', ''),
                'content_size': len(file_upload.get('content', ''))
            }
        return None

    def get_cached_request(self, request_hash):
        """Get cached request with comprehensive status"""
        if not self.cache_table:
            return None
            
        try:
            # Get main cache record
            cache_response = self.cache_table.get_item(Key={'request_hash': request_hash})
            
            if 'Item' not in cache_response:
                print(f"🔍 No cache entry found for {request_hash[:8]}")
                return None
                
            cache_item = cache_response['Item']
            
            # Check if cache is expired
            if self._is_cache_expired(cache_item):
                print(f"⏰ Cache expired for {request_hash[:8]}, cleaning up")
                self._cleanup_expired_cache(request_hash)
                return None
            
            status = cache_item['status']
            print(f"📋 Found cached request {request_hash[:8]} with status: {status}")
            
            result = {
                'status': status,
                'request_hash': request_hash,
                'created_at': float(cache_item.get('created_at', 0)),
                'updated_at': float(cache_item.get('updated_at', 0)),
                'payload': json.loads(cache_item.get('original_payload', '{}')),
                'estimated_duration': int(cache_item.get('estimated_duration', 60))
            }
            
            if status == 'completed':
                # Get full response data
                response_data = self._get_response_data(request_hash)
                if response_data:
                    result.update(response_data)
                else:
                    # Response data missing, mark as failed
                    print(f"❌ Response data missing for completed request {request_hash[:8]}")
                    self._update_cache_status(request_hash, 'failed', error='Response data not found')
                    result['status'] = 'failed'
                    result['error'] = 'Response data not found'
                    
            elif status == 'processing':
                # Get latest progress
                progress_data = self._get_progress_data(request_hash)
                if progress_data:
                    result.update(progress_data)
                else:
                    # Calculate progress based on time elapsed
                    result.update(self._calculate_time_based_progress(cache_item))
                    
            elif status == 'failed':
                result['error'] = cache_item.get('error', 'Unknown error occurred')
                
            return result
            
        except Exception as e:
            print(f"⚠️ Error retrieving cached request: {e}")
            return None

    def _is_cache_expired(self, cache_item):
        """Check if cache entry is expired (24 hours for completed, 6 hours for failed, 1 hour for stale processing)"""
        try:
            updated_at = float(cache_item.get('updated_at', 0))
            status = cache_item.get('status', '')
            current_time = time.time()
            
            if status == 'completed':
                return current_time - updated_at > (24 * 60 * 60)  # 24 hours
            elif status == 'failed':
                return current_time - updated_at > (6 * 60 * 60)   # 6 hours
            elif status == 'processing':
                return current_time - updated_at > (60 * 60)       # 1 hour (stale processing)
            else:
                return current_time - updated_at > (30 * 60)       # 30 minutes for unknown status
                
        except:
            return True  # Treat malformed entries as expired

    def _cleanup_expired_cache(self, request_hash):
        """Clean up expired cache entries"""
        try:
            # Delete from all tables
            if self.cache_table:
                self.cache_table.delete_item(Key={'request_hash': request_hash})
            if self.response_table:
                self.response_table.delete_item(Key={'request_hash': request_hash})
            if self.progress_table:
                self.progress_table.delete_item(Key={'request_hash': request_hash})
                
            print(f"🧹 Cleaned up expired cache for {request_hash[:8]}")
        except Exception as e:
            print(f"⚠️ Error cleaning expired cache: {e}")

    def _get_response_data(self, request_hash):
        """Get full response data from response table"""
        if not self.response_table:
            return None
        
        item = None
        try:
            response = self.response_table.get_item(Key={'request_hash': request_hash})
            if 'Item' in response:
                item = response['Item']
                return {
                    'response': item.get('response', ''),
                    'agent_type': item.get('agent_type', ''),
                    'processing_time': float(item.get('processing_time', 0)),
                    'image_urls': json.loads(item.get('image_urls', '[]')),
                    'audio_urls': json.loads(item.get('audio_urls', '[]')),
                    'files_created': json.loads(item.get('files_created', '[]')),
                    'tools_used': json.loads(item.get('tools_used', '[]')),
                    'metadata': json.loads(item.get('metadata', '{}'))
                }
            return None
        except Exception as e:
            print(f"⚠️ Error getting response data: {e}")
            return None

    def save_response(self, request_hash, result):
        """Save full response data"""
        if not self.response_table:
            print("⚠️ Response table not available")
            return
            
        try:
            current_time = time.time()
            ttl = int(current_time + (24 * 60 * 60))  # 24 hour TTL
            
            response_item = {
                'request_hash': request_hash,
                'response': str(result.get('response', ''))[:10000],  # Limit response size
                'agent_type': result.get('agent_type', ''),
                'processing_time': result.get('processing_time', 0),
                'image_urls': json.dumps(result.get('image_urls', []), default=str),
                'audio_urls': json.dumps(result.get('audio_urls', []), default=str),
                'files_created': json.dumps(result.get('files_created', []), default=str),
                'tools_used': json.dumps(result.get('strands_tools_available', []), default=str),
                'metadata': json.dumps({
                    'timestamp': result.get('timestamp', ''),
                    'success': True,
                    'cached_at': current_time
                }, default=str),
                'created_at': current_time,
                'ttl': ttl
            }
            
            self.response_table.put_item(Item=response_item)
            
            # Update main cache status
            self._update_cache_status(request_hash, 'completed', processing_time=result.get('processing_time', 0))
            
            # Final progress update
            self._update_progress(request_hash, 100, 'Request completed successfully')
            
            print(f"💾 Saved response for {request_hash[:8]}")
            
        except Exception as e:
            print(f"❌ Error saving response: {e}")
            self._update_cache_status(request_hash, 'failed', error=str(e))
        """Save error information"""
        try:
            error_msg = str(e)
            self._update_cache_status(request_hash, 'failed', error=error_msg)
            self._update_progress(request_hash, 0, f'Request failed: {error_msg[:100]}')
            
            print(f"❌ Saved error for {request_hash[:8]}: {error_msg[:100]}")
            
        except Exception as e:
            print(f"❌ Error saving error info: {e}")

    def _update_progress(self, request_hash, progress, message, phase=None):
        """Update progress information"""
        if not self.progress_table:
            return
            
        try:
            current_time = time.time()
            ttl = int(current_time + (6 * 60 * 60))  # 6 hour TTL for progress
            
            update_expression = "SET progress = :progress, progress_message = :message, updated_at = :updated_at, ttl = :ttl"
            expression_values = {
                ':progress': int(progress),
                ':message': str(message),
                ':updated_at': current_time,
                ':ttl': ttl
            }
            
            if phase:
                update_expression += ", current_phase = :phase"
                expression_values[':phase'] = phase
            
            self.progress_table.update_item(
                Key={'request_hash': request_hash},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ReturnValues="NONE"
            )
            
            print(f"📊 Progress {request_hash[:8]}: {progress}% - {message}")
            
        except Exception as e:
            print(f"⚠️ Error updating progress: {e}")

    def is_request_active(self, request_hash):
        """Check if request is currently being processed"""
        return request_hash in self.active_requests

    def mark_request_active(self, request_hash, thread_id):
        """Mark request as actively processing"""
        self.active_requests[request_hash] = {
            'thread_id': thread_id,
            'started_at': time.time(),
            'status': 'active'
        }

    def mark_request_inactive(self, request_hash):
        """Remove request from active processing"""
        if request_hash in self.active_requests:
            del self.active_requests[request_hash]

class EnhancedMultiAgentChatSystem:
    def __init__(self):
        """Initialize enhanced multi-agent chat system with advanced caching"""
        self.session = boto3.Session(region_name='us-east-1')
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        
        # Initialize enhanced cache manager
        self.cache_manager = EnhancedCacheManager(self.dynamodb)
        
        # Chat history table
        self.chat_table_name = 'workflow-agent-chat'
        self._ensure_chat_table_exists()
        
        # Agent cache
        self.agents = {}
        
        # ADD MARKETING CAMPAIGN MANAGER INITIALIZATION
        self.marketing_manager = MarketingCampaignManager(Environment.PRODUCTION)
        
        # Strands tools mapping
        self.strands_tools = {}
        
        # Add tools that were successfully imported
        if use_aws:
            self.strands_tools["use_aws"] = use_aws
        if speak:
            self.strands_tools["speak"] = speak
        if generate_image:
            self.strands_tools["generate_image"] = generate_image
        if nova_reels:
            self.strands_tools["nova_reels"] = nova_reels
        if image_reader:
            self.strands_tools["image_reader"] = image_reader
        if retrieve:
            self.strands_tools["retrieve"] = retrieve
        if memory:
            self.strands_tools["memory"] = memory
        if file_read:
            self.strands_tools["file_read"] = file_read
        if file_write:
            self.strands_tools["file_write"] = file_write
        if http_request:
            self.strands_tools["http_request"] = http_request
        if python_repl:
            self.strands_tools["python_repl"] = python_repl
        if calculator:
            self.strands_tools["calculator"] = calculator
        if current_time:
            self.strands_tools["current_time"] = current_time
        if workflow:
            self.strands_tools["workflow"] = workflow
        if use_llm:
            self.strands_tools["use_llm"] = use_llm
        
        print(f"🔧 Initialized with {len(self.strands_tools)} Strands tools: {list(self.strands_tools.keys())}")
        print(f"🎯 Marketing Campaign Manager integrated successfully")
        
    def _ensure_chat_table_exists(self):
        """Ensure chat history table exists"""
        try:
            self.chat_table = self.dynamodb.Table(self.chat_table_name)
            self.chat_table.load()
            print(f"✅ Using existing chat table: {self.chat_table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"🔧 Creating chat table: {self.chat_table_name}")
                self.chat_table = self.dynamodb.create_table(
                    TableName=self.chat_table_name,
                    KeySchema=[
                        {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'user_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                self.chat_table.wait_until_exists()
                print("✅ Chat table created successfully")
            else:
                self.chat_table = None

    def _is_long_running_request(self, payload):
        """UPDATED: Enhanced detection for requests that need background processing"""
        message = payload.get('message', '').lower()
        agent_type = payload.get('agent_type', '')
        context = payload.get('context', '').lower()
        
        # Length-based indicators
        message_length = len(message)
        context_length = len(context)
        
        # UPDATED: Add marketing campaign indicators
        marketing_indicators = [
            'marketing campaign' in message and ('generate' in message or 'create' in message),
            'customer list' in message or 'customer data' in message,
            'personalized message' in message and ('multiple' in message or 'batch' in message),
            'campaign generation' in message,
            'cold outreach' in message and ('customers' in message or 'prospects' in message),
            agent_type == 'marketing_campaign_agent' and message_length > 100,
            'csv' in context or 'spreadsheet' in context,
            context_length > 300  # Large customer data context
        ]
        
        # Content-based indicators for long-running operations
        long_running_indicators = [
            # Image/Video generation
            'generate image' in message and ('detailed' in message or 'complex' in message),
            'create image' in message and message_length > 100,
            'video' in message or 'nova_reels' in message,
            
            # Analysis and processing
            'comprehensive analysis' in message,
            'detailed report' in message,
            'analyze' in message and ('data' in message or 'file' in message),
            
            # Multiple operations
            message.count('and') >= 3,  # Multiple tasks
            message.count(',') >= 5,    # Complex instructions
            'workflow' in message and ('execute' in message or 'multiple' in message),
            
            # Time-intensive operations
            'convert' in message and 'speech' in message and message_length > 200,
            'transcribe' in message or 'translation' in message,
            'optimization' in message or 'training' in message,
            
            # Large content processing
            'process' in message and ('document' in message or 'file' in message),
            context_length > 500,  # Large context
            
            # Agent-specific operations
            agent_type in ['multimodal_agent', 'workflow_agent', 'analytics', 'marketing_campaign_agent'] and message_length > 150,
        ]
        
        # File upload indicator (always considered potentially long-running)
        has_file_upload = payload.get('file_upload') is not None
        
        # Complex agent types that typically take longer
        complex_agent_types = [
            'multimodal_agent',
            'workflow_agent', 
            'analytics',
            'aws_ai_agent',
            'marketing_campaign_agent'  # ADDED
        ]
        
        # Scoring system for long-running detection
        score = 0
        score += min(4, message_length // 150)  # 1 point per 150 chars, max 4
        score += min(3, context_length // 200)   # 1 point per 200 chars, max 3
        score += sum(long_running_indicators)    # 1 point per indicator
        score += sum(marketing_indicators)       # ADDED: Marketing indicators
        score += 3 if agent_type in complex_agent_types else 0
        score += 4 if has_file_upload else 0     # File uploads often take longer
        
        # Keywords that strongly suggest long processing
        if any(word in message for word in ['comprehensive', 'detailed analysis', 'multiple images', 'batch process', 'campaign generation']):
            score += 3
            
        print(f"📊 Long-running request scoring - Message: {message_length}chars, Agent: {agent_type}, Score: {score}/20")
        
        return score >= 6  # Threshold for background processing

    def _estimate_processing_duration(self, payload):
        """UPDATED: Estimate processing duration based on request complexity"""
        base_duration = 30  # Base 30 seconds
        
        message = payload.get('message', '').lower()
        agent_type = payload.get('agent_type', '')
        
        # Duration modifiers based on content
        duration_modifiers = {
            'image generation': 45,
            'video creation': 120,
            'speech synthesis': 30,
            'file processing': 60,
            'comprehensive analysis': 90,
            'workflow execution': 75,
            'multiple operations': 60,
            'marketing campaign': 120,  # ADDED
            'campaign generation': 90,  # ADDED
            'customer analysis': 60     # ADDED
        }
        
        for operation, additional_time in duration_modifiers.items():
            if operation.replace(' ', '') in message.replace(' ', ''):
                base_duration += additional_time
                
        # Agent-specific duration modifiers
        agent_duration_map = {
            'multimodal_agent': 60,
            'workflow_agent': 45,
            'analytics': 40,
            'aws_ai_agent': 35,
            'image_generator': 50,
            'marketing_campaign_agent': 90  # ADDED
        }
        
        if agent_type in agent_duration_map:
            base_duration += agent_duration_map[agent_type]
            
        # File upload adds processing time
        if payload.get('file_upload'):
            base_duration += 45
            
        # Context complexity (especially for customer data)
        context_length = len(payload.get('context', ''))
        if context_length > 500:
            base_duration += min(60, context_length // 100)
            
        return min(base_duration, 300)  # Cap at 5 minutes

    def process_chat_request(self, payload):
        """Main method to process chat requests with enhanced caching"""
        request_hash = self.cache_manager._generate_request_hash(payload)
        
        print(f"🔄 Processing request {request_hash[:8]} for user {payload.get('user_id', 'unknown')}")
        
        # Check cache first
        cached_request = self.cache_manager.get_cached_request(request_hash)
        
        if cached_request:
            status = cached_request['status']
            
            if status == 'completed':
                print(f"✅ Returning completed cached result for {request_hash[:8]}")
                return {
                    'success': True,
                    'status': 'completed',
                    'request_hash': request_hash,
                    'from_cache': True,
                    **{k: v for k, v in cached_request.items() 
                       if k not in ['status', 'request_hash']}
                }
                
            elif status == 'processing':
                print(f"⏳ Request {request_hash[:8]} is already processing")
                return {
                    'success': True,
                    'status': 'processing',
                    'request_hash': request_hash,
                    'progress': cached_request.get('progress', 0),
                    'message': cached_request.get('progress_message', 'Processing your request...'),
                    'estimated_completion': cached_request.get('estimated_completion', time.time() + 60),
                    'poll_url': f"/poll/{request_hash}",
                    'poll_interval': 5,
                    'from_cache': True
                }
                
            elif status == 'failed':
                print(f"❌ Request {request_hash[:8]} previously failed")
                # For failed requests, we might want to retry automatically or return the cached error
                if time.time() - cached_request.get('updated_at', 0) > 300:  # 5 minutes
                    print(f"🔄 Retrying previously failed request {request_hash[:8]}")
                    # Continue to process as new request
                else:
                    return {
                        'success': False,
                        'status': 'failed',
                        'request_hash': request_hash,
                        'error': cached_request.get('error', 'Unknown error occurred'),
                        'from_cache': True
                    }
        
        # Determine if this is a long-running request
        is_long_running = self._is_long_running_request(payload)
        estimated_duration = self._estimate_processing_duration(payload)
        
        if is_long_running:
            print(f"🔄 Long-running request detected, processing in background: {request_hash[:8]}")
            
            # Create cache entry for background processing
            self.cache_manager.create_cache_entry(
                request_hash, 
                payload, 
                status='processing', 
                estimated_duration=estimated_duration
            )
            
            # Start background processing
            thread = Thread(target=self._process_in_background, args=(payload, request_hash))
            thread.daemon = True
            thread.start()
            
            # Mark as actively processing
            self.cache_manager.mark_request_active(request_hash, thread.ident)
            
            return {
                'success': True,
                'status': 'processing',
                'request_hash': request_hash,
                'message': f'Long-running request detected. Processing in background.',
                'progress': 0,
                'estimated_duration': estimated_duration,
                'estimated_completion': time.time() + estimated_duration,
                'poll_url': f"/poll/{request_hash}",
                'poll_interval': 5,
                'from_cache': False
            }
        
        # Process short requests immediately
        print(f"⚡ Processing short request immediately: {request_hash[:8]}")
        
        # Create cache entry for immediate processing
        self.cache_manager.create_cache_entry(request_hash, payload, status='processing', estimated_duration=30)
        
        try:
            result = self._process_message_immediately(payload)
            
            if result['success']:
                # Save successful result
                self.cache_manager.save_response(request_hash, result)
                
                return {
                    'success': True,
                    'status': 'completed',
                    'request_hash': request_hash,
                    'from_cache': False,
                    **{k: v for k, v in result.items() if k != 'success'}
                }
            else:
                # Save error
                self.cache_manager.save_error(request_hash, result.get('error'))
                
                return {
                    'success': False,
                    'status': 'failed',
                    'request_hash': request_hash,
                    'error': result.get('error'),
                    'from_cache': False
                }
                
        except Exception as e:
            print(f"❌ Error processing immediate request {request_hash[:8]}: {e}")
            self.cache_manager.save_error(request_hash, str(e))
            
            return {
                'success': False,
                'status': 'failed',
                'request_hash': request_hash,
                'error': str(e),
                'from_cache': False
            }

    def _process_in_background(self, payload, request_hash):
        """Enhanced background processing with comprehensive progress tracking"""
        try:
            print(f"🔄 Starting background processing for {request_hash[:8]}")
            
            # Phase 1: Initialization (0-10%)
            self.cache_manager._update_progress(
                request_hash, 5, "Initializing agent and loading tools...", "initialization"
            )
            
            # Get user configuration and determine agent
            user_id = payload.get('user_id')
            workflow_data = self.fetch_workflow_config(user_id)
            agent_type = payload.get('agent_type') or self.determine_agent_type(payload.get('message'), workflow_data)
            
            # Phase 2: Agent Setup (10-20%)
            self.cache_manager._update_progress(
                request_hash, 15, f"Setting up {agent_type} agent with required tools...", "agent_setup"
            )
            
            agent = self.get_or_create_agent(user_id, agent_type)
            
            # Phase 3: Pre-processing (20-30%)
            self.cache_manager._update_progress(
                request_hash, 25, "Processing request parameters and context...", "preprocessing"
            )
            
            # Handle file upload if present
            file_url = None
            if payload.get('file_upload'):
                self.cache_manager._update_progress(
                    request_hash, 30, "Processing uploaded file...", "file_processing"
                )
                file_url = self.process_file_upload(user_id, payload['file_upload'])
            
            # Phase 4: Main Processing (30-85%)
            self.cache_manager._update_progress(
                request_hash, 35, "Processing your request with AI agent...", "main_processing"
            )
            
            # Simulate incremental progress during processing
            progress_thread = Thread(
                target=self._simulate_detailed_progress, 
                args=(request_hash, 35, 85, 10)  # From 35% to 85% over ~10 intervals
            )
            progress_thread.daemon = True
            progress_thread.start()
            
            # Actually process the message
            start_time = time.time()
            result = self._process_message_with_agent(agent, payload, file_url)
            processing_time = time.time() - start_time
            
            # Phase 5: Post-processing (85-95%)
            self.cache_manager._update_progress(
                request_hash, 90, "Finalizing response and saving results...", "postprocessing"
            )
            
            if result['success']:
                result['processing_time'] = processing_time
                result['request_hash'] = request_hash
                result['background_processed'] = True
                
                # Phase 6: Completion (95-100%)
                self.cache_manager._update_progress(
                    request_hash, 95, "Saving response to cache...", "completion"
                )
                
                # Save successful result
                self.cache_manager.save_response(request_hash, result)
                
                print(f"✅ Background processing completed successfully for {request_hash[:8]}")
                
            else:
                # Save error result
                self.cache_manager.save_error(request_hash, result.get('error', 'Unknown error'))
                print(f"❌ Background processing failed for {request_hash[:8]}: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Background processing exception for {request_hash[:8]}: {e}")
            import traceback
            traceback.print_exc()
            
            self.cache_manager.save_error(request_hash, str(e))
            
        finally:
            # Clean up
            self.cache_manager.mark_request_inactive(request_hash)

    def _simulate_detailed_progress(self, request_hash, start_progress, end_progress, steps):
        """Simulate detailed progress updates during main processing"""
        progress_messages = [
            "Analyzing request parameters...",
            "Loading AI models and tools...", 
            "Processing with neural networks...",
            "Generating content...",
            "Applying transformations...",
            "Running quality checks...",
            "Optimizing output...",
            "Preparing final response...",
            "Validating results...",
            "Almost complete..."
        ]
        
        step_size = (end_progress - start_progress) / steps
        step_duration = 8  # seconds per step
        
        for i in range(steps):
            time.sleep(step_duration)
            
            current_progress = int(start_progress + (i * step_size))
            message_idx = min(i, len(progress_messages) - 1)
            
            try:
                self.cache_manager._update_progress(
                    request_hash,
                    current_progress,
                    progress_messages[message_idx],
                    "main_processing"
                )
            except:
                break  # Stop if progress update fails

    def poll_request_status(self, request_hash):
        """Poll the status of a processing request"""
        print(f"🔍 Polling status for request {request_hash[:8]}")
        
        cached_request = self.cache_manager.get_cached_request(request_hash)
        
        if not cached_request:
            return {
                'success': False,
                'status': 'not_found',
                'error': 'Request not found, expired, or never existed',
                'request_hash': request_hash
            }
        
        status = cached_request['status']
        
        if status == 'completed':
            print(f"✅ Request {request_hash[:8]} completed")
            return {
                'success': True,
                'status': 'completed',
                'request_hash': request_hash,
                **{k: v for k, v in cached_request.items() 
                   if k not in ['status', 'request_hash']}
            }
            
        elif status == 'processing':
            elapsed_time = int(time.time() - cached_request.get('created_at', time.time()))
            estimated_remaining = max(0, cached_request.get('estimated_duration', 60) - elapsed_time)
            
            return {
                'success': True,
                'status': 'processing',
                'request_hash': request_hash,
                'progress': cached_request.get('progress', 0),
                'message': cached_request.get('progress_message', 'Processing your request...'),
                'current_phase': cached_request.get('current_phase', 'processing'),
                'elapsed_time': elapsed_time,
                'estimated_remaining': estimated_remaining,
                'estimated_completion': cached_request.get('estimated_completion', time.time() + 60),
                'poll_interval': 5
            }
            
        else:  # failed
            return {
                'success': False,
                'status': 'failed',
                'request_hash': request_hash,
                'error': cached_request.get('error', 'Unknown error occurred'),
                'elapsed_time': int(time.time() - cached_request.get('created_at', time.time()))
            }

    def _process_message_immediately(self, payload):
        """Process message immediately for short requests"""
        try:
            user_id = payload.get('user_id')
            
            # Get workflow config and agent
            workflow_data = self.fetch_workflow_config(user_id)
            agent_type = payload.get('agent_type') or self.determine_agent_type(payload.get('message'), workflow_data)
            agent = self.get_or_create_agent(user_id, agent_type)
            
            # Handle file upload if present
            file_url = None
            if payload.get('file_upload'):
                file_url = self.process_file_upload(user_id, payload['file_upload'])
            
            # Process with agent
            return self._process_message_with_agent(agent, payload, file_url)
            
        except Exception as e:
            print(f"❌ Error in immediate processing: {e}")
            return {'success': False, 'error': str(e)}

    def _process_message_with_agent(self, agent, payload, file_url=None):
        """Process message with the given agent"""
        try:
            user_id = payload.get('user_id')
            message = payload.get('message')
            context = payload.get('context')
            
            start_time = time.time()
            
            # Enhance message with context and file
            enhanced_message = self._enhance_message(user_id, message, context, file_url)
            
            # Process with agent
            response = agent(enhanced_message)
            response_text = self._extract_response_text(response)
            
            # Extract various content types from response
            image_urls = self._extract_content_urls(response, 'image')
            audio_urls = self._extract_content_urls(response, 'audio')
            files_created = self._extract_content_urls(response, 'file')
            
            # Save interaction to chat history
            self.save_chat_message(user_id, message, response_text, payload.get('agent_type', 'unknown'))
            
            processing_time = time.time() - start_time
            
            result = {
                'response': response_text,
                'agent_type': payload.get('agent_type', 'unknown'),
                'processing_time': processing_time,
                'strands_tools_available': list(self.strands_tools.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add content URLs if present
            if image_urls:
                result['image_urls'] = image_urls
            if audio_urls:
                result['audio_urls'] = audio_urls
            if files_created:
                result['files_created'] = files_created
            
            return {'success': True, **result}
            
        except Exception as e:
            print(f"❌ Error processing message with agent: {e}")
            return {'success': False, 'error': str(e)}

    def _enhance_message(self, user_id, message, context=None, file_url=None):
        """Enhance message with context, history, and file information"""
        enhanced_parts = []
        
        # Add context if provided
        if context:
            enhanced_parts.append(f"Context: {context}")
        
        # Add file information if provided
        if file_url:
            enhanced_parts.append(f"Uploaded file URL: {file_url}")
        
        # Add recent chat history for continuity
        chat_history = self.get_chat_history(user_id, limit=3)
        if chat_history:
            history_context = "\n".join([
                f"Previous: {item['message'][:100]} -> {str(item['response'])[:100]}..."
                for item in reversed(chat_history)
            ])
            enhanced_parts.append(f"Recent conversation history:\n{history_context}")
        
        # Add current message
        enhanced_parts.append(f"Current request: {message}")
        
        return "\n\n".join(enhanced_parts)

    # Keep all existing methods from your original code...
    def fetch_workflow_config(self, user_id):
        """Fetch workflow configuration from S3"""
        try:
            bucket_name = 'qubitz-customer-prod'
            
            # Try user-specific config first
            try:
                key = f'{user_id}/workflow.json'
                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                workflow_data = json.loads(response['Body'].read().decode('utf-8'))
                print(f"✅ User-specific workflow config loaded for {user_id}")
                return workflow_data
            except:
                # Try project-specific config
                try:
                    objects = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f'{user_id}/')
                    if 'Contents' in objects and objects['Contents']:
                        for obj in objects['Contents']:
                            if obj['Key'].endswith('.json'):
                                key = obj['Key']
                                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                                workflow_data = json.loads(response['Body'].read().decode('utf-8'))
                                print(f"✅ Project workflow config loaded for {user_id}: {key}")
                                return workflow_data
                except:
                    pass
            
            print(f"⚠️ No workflow config found for {user_id}")
            return None
            
        except Exception as e:
            print(f"⚠️ Error fetching workflow config: {e}")
            return None

    def determine_agent_type(self, message, workflow_data=None):
        """UPDATED: Determine which agent type to use based on message content"""
        message_lower = message.lower()
        
        # UPDATED: Add marketing campaign detection (NEW)
        if any(word in message_lower for word in [
            'marketing campaign', 'cold outreach', 'lead generation', 'customer outreach',
            'marketing message', 'campaign generation', 'personalized message', 'b2b outreach',
            'sales campaign', 'customer campaign', 'marketing automation', 'cold messaging',
            'prospect outreach', 'email campaign'
        ]):
            return 'marketing_campaign_agent'
        
        # Enhanced agent type detection
        elif any(word in message_lower for word in ['generate image', 'create image', 'make picture', 'draw', 'illustrate']):
            return 'image_generator'
        elif any(word in message_lower for word in ['speak', 'voice', 'audio', 'tts', 'text to speech', 'polly']):
            return 'aws_ai_agent'
        elif any(word in message_lower for word in ['video', 'nova_reels', 'reel', 'animation']):
            return 'multimodal_agent'
        elif any(word in message_lower for word in ['sentiment', 'analyze text', 'comprehend', 'translate', 'aws']):
            return 'aws_ai_agent'
        elif any(word in message_lower for word in ['workflow', 'orchestrate', 'process', 'coordinate', 'chain']):
            return 'workflow_agent'
        elif any(word in message_lower for word in ['validate', 'check', 'verify', 'review', 'compliance']):
            return 'validator'
        elif any(word in message_lower for word in ['analyze', 'analytics', 'report', 'metrics', 'calculate', 'math']):
            return 'analytics'
        elif any(word in message_lower for word in ['comprehensive', 'detailed', 'multiple', 'complex']):
            return 'multimodal_agent'
        else:
            return 'ai_chat_agent'

    def create_agent_from_workflow(self, workflow_data, agent_type):
        """UPDATED: Create specialized agent based on workflow configuration"""
        system_prompts = {
            # UPDATED: Add marketing campaign agent (NEW)
            'marketing_campaign_agent': f"""
You are an expert B2B marketing campaign specialist integrated with advanced AI tools.

MARKETING CAPABILITIES:
- Generate personalized cold outreach messages
- Create industry-specific messaging
- Develop customer segmentation strategies  
- Analyze customer data for lead scoring
- Create multi-channel campaign content

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

MARKETING PRINCIPLES:
- Always create truthful, non-misleading content
- Personalize based on industry and role
- Focus on value proposition, not hard selling
- Ask thoughtful questions to engage prospects
- Respect professional communication standards

When users request marketing campaigns:
1. Parse customer data from context (name, company, position, email)
2. Analyze industry and seniority level for each customer
3. Generate personalized messages using appropriate tone
4. Create both quick-connect (50-75 words) and detailed (175-200 words) versions
5. Ensure messages are truthful and professional

CRITICAL: Never invent statistics or fake achievements. Use phrases like "designed to help" instead of definitive claims.

Be professional, strategic, and results-focused while maintaining ethical standards.
            """,
            
            'ai_chat_agent': f"""
You are an intelligent customer support chat agent with access to advanced tools and AWS services.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Provide helpful, professional responses and use appropriate tools when beneficial.
Be conversational and ask clarifying questions when needed.
            """,
            
            'ai_text_generator': f"""
You are an AI text generation specialist capable of creating high-quality content.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Generate professional, well-structured content based on user requirements.
Use tools like file_write to save content when requested.
            """,
            
            'validator': f"""
You are a validation specialist with advanced analysis capabilities.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Validate data, check compliance, and provide comprehensive feedback.
Use python_repl for computational validation when needed.
            """,
            
            'analytics': f"""
You are an analytics specialist with computational and AI-powered insights.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Perform data analysis, create reports, and provide actionable insights.
Use calculator and python_repl for mathematical operations.
            """,
            
            'image_generator': f"""
You are an AI image generation and analysis specialist.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Create images with generate_image, analyze with image_reader, and explain your process.
Provide detailed descriptions of generated content.
            """,
            
            'aws_ai_agent': f"""
You are an AWS AI services coordinator with comprehensive cloud AI capabilities.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Use AWS services via use_aws tool for Polly (TTS), Comprehend (sentiment), 
Textract (documents), Rekognition (images), and Translate.
            """,
            
            'multimodal_agent': f"""
You are a multimodal AI agent capable of handling text, images, video, and audio.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Combine multiple tools to create rich, multimodal responses.
Generate images, videos, audio, and text as needed.
            """,
            
            'workflow_agent': f"""
You are a workflow orchestration agent managing complex processes.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Orchestrate complex workflows using multiple tools in sequence.
Break down complex tasks into manageable steps.
            """
        }
        
        system_prompt = system_prompts.get(agent_type, system_prompts['ai_chat_agent'])
        
        # UPDATED: Use environment-specific model selection
        model_config = self.marketing_manager.current_model if agent_type == 'marketing_campaign_agent' else None
        
        if model_config:
            # Use marketing-specific model configuration
            bedrock_model = BedrockModel(
                model_id=model_config.model_id,
                boto_session=self.session,
                max_tokens=4000,
                params={"temperature": 0.3, "top_p": 0.9}
            )
        else:
            # Use default model for non-marketing agents
            bedrock_model = BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                boto_session=self.session,
                max_tokens=4000,
                params={"temperature": 0.3, "top_p": 0.9}
            )
        
        # Get tools for this agent
        agent_tools = self._get_agent_tools(agent_type)
        
        return Agent(
            model=bedrock_model,
            conversation_manager=SlidingWindowConversationManager(window_size=5),
            system_prompt=system_prompt,
            tools=agent_tools
        )

    def _get_agent_tools(self, agent_type):
        """UPDATED: Get tools for specific agent type"""
        agent_tool_mappings = {
            # UPDATED: Add marketing campaign agent tools (NEW)
            'marketing_campaign_agent': [
                "use_aws",          # For AWS AI services (Comprehend, Translate, etc.)  
                "http_request",     # For web research and API calls
                "file_read",        # For processing customer CSV files
                "file_write",       # For saving campaign results
                "python_repl",      # For data analysis and processing
                "calculator",       # For campaign metrics
                "current_time",     # For scheduling
                "memory"            # For campaign context
            ],
            
            'ai_chat_agent': ["use_aws", "speak", "memory", "calculator", "current_time"],
            'ai_text_generator': ["use_aws", "speak", "file_write", "file_read"],
            'validator': ["use_aws", "python_repl", "file_read", "calculator"],
            'analytics': ["python_repl", "calculator", "use_aws", "file_read", "file_write"],
            'image_generator': ["generate_image", "image_reader", "use_aws"],
            'aws_ai_agent': ["use_aws", "speak", "image_reader", "memory"],
            'multimodal_agent': ["generate_image", "nova_reels", "image_reader", "speak", "use_aws"],
            'workflow_agent': ["workflow", "use_llm", "use_aws", "python_repl"],
        }
        
        tool_names = agent_tool_mappings.get(agent_type, ["use_aws", "calculator"])
        
        tools = []
        for tool_name in tool_names:
            if tool_name in self.strands_tools:
                tools.append(self.strands_tools[tool_name])
        
        print(f"🤖 Agent {agent_type} initialized with {len(tools)} tools")
        return tools

    def get_or_create_agent(self, user_id, agent_type):
        """Get or create agent for user and type"""
        agent_key = f"{user_id}_{agent_type}"
        
        if agent_key not in self.agents:
            print(f"🤖 Creating new agent: {agent_type} for user {user_id}")
            workflow_data = self.fetch_workflow_config(user_id)
            self.agents[agent_key] = self.create_agent_from_workflow(workflow_data, agent_type)
            
        return self.agents[agent_key]

    def save_chat_message(self, user_id, message, response, agent_type):
        """Save chat interaction to DynamoDB"""
        if not self.chat_table:
            return
            
        try:
            timestamp = int(time.time() * 1000)
            
            self.chat_table.put_item(Item={
                'user_id': user_id,
                'timestamp': timestamp,
                'message': message[:1000],  # Truncate if too long
                'response': str(response)[:1000],  # Truncate if too long
                'agent_type': agent_type,
                'created_at': datetime.now().isoformat()
            })
            
            print(f"💾 Saved chat interaction for {user_id}")
        except Exception as e:
            print(f"⚠️ Error saving chat message: {e}")

    def get_chat_history(self, user_id, limit=10):
        """Get recent chat history for user"""
        if not self.chat_table:
            return []
            
        try:
            response = self.chat_table.query(
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={':user_id': user_id},
                ScanIndexForward=False,
                Limit=limit
            )
            
            return response.get('Items', [])
        except Exception as e:
            print(f"⚠️ Error getting chat history: {e}")
            return []

    def process_file_upload(self, user_id, file_upload):
        """Process file upload and store in S3"""
        try:
            if isinstance(file_upload, dict):
                file_name = file_upload.get('filename')
                file_content = base64.b64decode(file_upload.get('content'))
                content_type = file_upload.get('contentType', 'application/octet-stream')
            else:
                print("⚠️ Invalid file_upload format")
                return None
            
            file_id = str(uuid.uuid4())
            bucket_name = 'qubitz-customer-prod'
            key = f'{user_id}/files/{file_id}_{file_name}'
            
            self.s3_client.put_object(
                Body=file_content,
                Bucket=bucket_name,
                Key=key,
                ContentType=content_type
            )
            
            file_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': key},
                ExpiresIn=3600
            )
            
            print(f"📁 File uploaded: {key}")
            return file_url
            
        except Exception as e:
            print(f"❌ Error processing file upload: {e}")
            return None

    def _extract_response_text(self, response):
        """Extract text from agent response"""
        try:
            if hasattr(response, 'content'):
                if isinstance(response.content, list):
                    text_parts = []
                    for item in response.content:
                        if isinstance(item, dict):
                            if 'text' in item:
                                text_parts.append(item['text'])
                            elif 'content' in item:
                                text_parts.append(str(item['content']))
                        else:
                            text_parts.append(str(item))
                    return ' '.join(text_parts)
                else:
                    return str(response.content)
            else:
                return str(response)
        except Exception as e:
            print(f"⚠️ Error extracting response: {e}")
            return str(response)
    
    def _extract_content_urls(self, response, content_type):
        """Extract URLs from response for specific content type"""
        try:
            urls = []
            if hasattr(response, 'content') and isinstance(response.content, list):
                for item in response.content:
                    if isinstance(item, dict):
                        if f'{content_type}_url' in item:
                            urls.append(item[f'{content_type}_url'])
                        elif content_type in item and isinstance(item[content_type], dict):
                            if 'url' in item[content_type]:
                                urls.append(item[content_type]['url'])
                            elif 'source' in item[content_type]:
                                urls.append(item[content_type]['source'])
            return urls if urls else None
        except Exception as e:
            print(f"⚠️ Error extracting {content_type} URLs: {e}")
            return None

# Global chat system instance
chat_system = None

def lambda_handler(event, context):
    """UPDATED: Enhanced Lambda handler with comprehensive caching and polling support"""
    global chat_system
    
    try:
        # Initialize system
        if chat_system is None:
            print("🔧 Initializing Enhanced MultiAgent Chat System with Marketing Campaigns...")
            chat_system = EnhancedMultiAgentChatSystem()
            print("✅ Enhanced MultiAgent Chat System with Marketing Campaigns initialized successfully")
        
        # Parse HTTP method and path
        http_method = event.get('httpMethod', 'POST')
        path = event.get('path', '')
        
        # Handle polling requests (GET /poll/{request_hash})
        if http_method == 'GET' and '/poll/' in path:
            request_hash = path.split('/poll/')[-1]
            print(f"🔍 Polling request for hash: {request_hash[:8]}")
            
            result = chat_system.poll_request_status(request_hash)
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps(result, default=str)
            }
        
        # Handle OPTIONS requests for CORS
        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }
        
        # Handle chat requests (POST)
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Validate required fields
        user_id = body.get('user_id')
        message = body.get('message')
        
        if not user_id or not message:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': 'Missing required fields: user_id and message',
                    'required_fields': ['user_id', 'message'],
                    'optional_fields': ['context', 'agent_type', 'file_upload'],
                    'supported_agent_types': [
                        'ai_chat_agent', 'ai_text_generator', 'validator',
                        'analytics', 'image_generator', 'aws_ai_agent',
                        'multimodal_agent', 'workflow_agent', 'marketing_campaign_agent'  # ADDED
                    ],
                    'available_tools': list(chat_system.strands_tools.keys()),
                    'marketing_features': {  # ADDED
                        'campaign_generation': 'Generate personalized B2B marketing messages',
                        'industry_targeting': 'Technology, Healthcare, Finance, Retail, Manufacturing, Education, Consulting',
                        'seniority_levels': 'C-Level, Director, Manager, Specialist, Junior',
                        'message_types': 'Quick Connect (50-75 words), Value Proposition (175-200 words)',
                        'truthful_messaging': 'No fake statistics or unverifiable claims'
                    },
                    'caching_info': {
                        'automatic_deduplication': 'Identical requests return cached results',
                        'background_processing': 'Long requests processed asynchronously',
                        'progress_tracking': 'Real-time progress updates available',
                        'cache_duration': '24 hours for completed, 6 hours for failed'
                    },
                    'polling_info': {
                        'poll_endpoint': '/poll/{request_hash}',
                        'poll_interval': '5 seconds recommended',
                        'long_running_threshold': 'Automatically detected'
                    }
                })
            }
        
        print(f"💬 Processing chat request for user: {user_id}")
        print(f"📝 Message preview: {message[:100]}{'...' if len(message) > 100 else ''}")
        print(f"🎯 Agent type: {body.get('agent_type', 'auto-detect')}")
        
        # Process the chat request with enhanced caching
        result = chat_system.process_chat_request(body)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result, default=str)
        }
        
    except Exception as e:
        print(f"❌ Lambda error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'message': 'Internal server error occurred. Please try again or contact support.'
            })
        }

# UPDATED: Enhanced test and sample payloads with marketing examples
if __name__ == "__main__":
    print("=== Enhanced Multi-Agent Chat System with Marketing Campaigns ===")
    print("Features:")
    print("✅ Request deduplication with deterministic hashing")
    print("✅ Automatic background processing for long requests") 
    print("✅ Real-time progress tracking with detailed phases")
    print("✅ Comprehensive caching with TTL management")
    print("✅ Intelligent retry logic for failed requests")
    print("✅ Multi-table storage for optimal performance")
    print("✅ Marketing campaign generation with industry targeting")  # ADDED
    print("✅ Truthful messaging without fake statistics")  # ADDED
    
    # UPDATED: Sample test payloads with marketing examples
    sample_payloads = {
        'simple_chat': {
            "user_id": "marketing_manager_001",
            "message": "Hello! Can you help me with customer outreach strategies?",
            "agent_type": "ai_chat_agent"
        },
        
        # NEW: Marketing campaign examples
        'marketing_campaign_basic': {
            "user_id": "marketing_manager_001",
            "message": "Generate personalized marketing campaign messages for my customer list",
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
        },
        
        'marketing_campaign_comprehensive': {
            "user_id": "marketing_manager_001",
            "message": "Create comprehensive B2B marketing campaign with personalized messages for multiple customer segments across different industries",
            "agent_type": "marketing_campaign_agent",
            "context": """Large customer database with 50+ prospects:
- Technology: CTOs, Engineering Directors, Software Managers
- Healthcare: Chief Medical Officers, IT Directors, Operations Managers  
- Finance: CFOs, Risk Management Directors, Compliance Managers

Our SaaS Platform:
- AI-powered analytics and reporting
- Regulatory compliance automation
- Real-time data visualization
- Enterprise security features
- Investment: $25,000 - $100,000
- ROI within 6-12 months

Need both quick-connect and detailed value proposition messages for each segment."""
        },
        
        'existing_analytics': {
            "user_id": "marketing_manager_001", 
            "message": "Analyze this customer data and provide insights on lead generation opportunities for our SaaS platform targeting marketing managers",
            "agent_type": "analytics",
            "context": "Company: Tech Solutions Inc, Industry: Software, Size: 50-200 employees, Previous engagement: Downloaded whitepaper"
        },
        
        'long_running_comprehensive': {
            "user_id": "marketing_manager_001",
            "message": "Create a comprehensive marketing campaign analysis including: customer sentiment analysis, generate promotional images, convert key messages to speech, create a video presentation, and provide detailed analytics report with projections",
            "agent_type": "multimodal_agent",
            "context": "Q3 2024 campaign for new product launch targeting B2B customers"
        },
        
        'duplicate_request_test': {
            "user_id": "marketing_manager_001",
            "message": "Hello! Can you help me with customer outreach strategies?",
            "agent_type": "ai_chat_agent"
            # This duplicates the first request to test caching
        }
    }
    
    # Test caching behavior
    print("\n=== Testing Marketing Campaign Integration ===")
    
    # Test 1: Simple request (should process immediately)
    print("\n1. Testing simple request (immediate processing):")
    test_event = {
        "httpMethod": "POST",
        "body": json.dumps(sample_payloads['simple_chat'])
    }
    
    result = lambda_handler(test_event, None)
    print(f"Status: {result['statusCode']}")
    
    if result['statusCode'] == 200:
        response_data = json.loads(result['body'])
        print(f"Success: {response_data.get('success')}")
        print(f"Status: {response_data.get('status')}")
        print(f"From Cache: {response_data.get('from_cache', False)}")
        print(f"Request Hash: {response_data.get('request_hash', '')[:8]}...")
    
    # Test 2: Marketing campaign (should go to background)
    print("\n2. Testing marketing campaign request (background processing):")
    test_event_marketing = {
        "httpMethod": "POST", 
        "body": json.dumps(sample_payloads['marketing_campaign_basic'])
    }
    
    result_marketing = lambda_handler(test_event_marketing, None)
    
    if result_marketing['statusCode'] == 200:
        response_data = json.loads(result_marketing['body'])
        print(f"Status: {response_data.get('status')}")
        print(f"Agent Type Detected: {response_data.get('agent_type', 'N/A')}")
        print(f"Request Hash: {response_data.get('request_hash', '')[:8]}...")
        print(f"Poll URL: {response_data.get('poll_url')}")
        print(f"Estimated Duration: {response_data.get('estimated_duration')}s")
        
        # Test polling
        if response_data.get('status') == 'processing':
            request_hash = response_data['request_hash']
            print(f"\n3. Testing polling for marketing campaign {request_hash[:8]}:")
            
            poll_event = {
                "httpMethod": "GET",
                "path": f"/poll/{request_hash}"
            }
            
            poll_result = lambda_handler(poll_event, None)
            if poll_result['statusCode'] == 200:
                poll_data = json.loads(poll_result['body'])
                print(f"Poll Status: {poll_data.get('status')}")
                print(f"Progress: {poll_data.get('progress', 0)}%")
                print(f"Phase: {poll_data.get('current_phase', 'unknown')}")
                print(f"Message: {poll_data.get('message', 'No message')}")
    
    print(f"\n🎯 MARKETING CAMPAIGN FEATURES SUMMARY:")
    print("""
    ✅ MARKETING CAMPAIGN AGENT
    - Automatic detection of marketing campaign requests
    - Industry-specific messaging (Technology, Healthcare, Finance, etc.)
    - Seniority-level targeting (C-Level, Director, Manager, Specialist, Junior)
    - Two message types: Quick Connect (50-75 words) & Value Proposition (175-200 words)
    
    ✅ TRUTHFUL MESSAGING FRAMEWORK
    - No fake statistics or unverifiable claims
    - Uses phrases like "designed to help" instead of "guaranteed results"
    - Professional, consultative tone
    - Industry-appropriate language and pain points
    
    ✅ BATCH PROCESSING CAPABILITY
    - Handle multiple customers in single request
    - Background processing for large customer lists
    - Real-time progress tracking during generation
    - Automatic caching to prevent duplicate processing
    
    ✅ INTEGRATION WITH EXISTING SYSTEM
    - Same caching and polling infrastructure
    - Compatible with existing agent framework
    - Uses all available Strands tools
    - Maintains existing API format
    """)
    
    print("\n=== Ready for Marketing Campaign Manager Integration! ===")

# UPDATED: Helper function for creating marketing campaign requests
def create_marketing_campaign_request(customers: List[dict], offering_details: dict) -> dict:
    """Helper function to create marketing campaign request payload"""
    
    # Convert customer list to context string for the agent
    customer_context = "Customer Data:\n"
    for i, customer in enumerate(customers, 1):
        customer_context += f"{i}. {customer.get('first_name', '')} {customer.get('last_name', '')}"
        if customer.get('company'):
            customer_context += f" at {customer['company']}"
        if customer.get('position'):
            customer_context += f" ({customer['position']})"
        if customer.get('email'):
            customer_context += f" - {customer['email']}"
        customer_context += "\n"
    
    # Create offering context
    offering_context = f"""
Offering Details:
- Name: {offering_details.get('name', 'Not specified')}
- Objective: {offering_details.get('objective', 'Not specified')}
- Benefits: {offering_details.get('benefits', 'Not specified')}
- Target Industry: {offering_details.get('target_industry', 'Not specified')}
- Price Range: {offering_details.get('price_range', 'Not specified')}
- Duration: {offering_details.get('duration', 'Not specified')}
    """
    
    return {
        "user_id": "marketing_campaign_manager",
        "message": (
            "Generate personalized B2B marketing campaign messages for the following customers. "
            "Create both quick-connect (50-75 words) and value proposition (175-200 words) messages for each customer. "
            "Ensure messages are truthful, industry-appropriate, and role-specific."
        ),
        "agent_type": "marketing_campaign_agent",
        "context": customer_context + offering_context
    }

    def _get_progress_data(self, request_hash):
        """Get latest progress data from progress table"""
        if not self.progress_table:
            return None
            
        try:
            response = self.progress_table.get_item(Key={'request_hash': request_hash})
            if 'Item' in response:
                item = response['Item']
                return {
                    'progress': int(item.get('progress', 0)),
                    'progress_message': item.get('progress_message', 'Processing...'),
                    'current_phase': item.get('current_phase', 'initialization'),
                    'phases_completed': json.loads(item.get('phases_completed', '[]')),
                    'estimated_completion': float(item.get('estimated_completion', 0))
                }
            return None
        except Exception as e:
            print(f"⚠️ Error getting progress data: {e}")
            return None

    def _calculate_time_based_progress(self, cache_item):
        """Calculate progress based on elapsed time"""
        try:
            created_at = float(cache_item.get('created_at', time.time()))
            estimated_duration = int(cache_item.get('estimated_duration', 60))
            elapsed = time.time() - created_at
            
            # Non-linear progress curve
            if elapsed < estimated_duration * 0.2:
                progress = int((elapsed / (estimated_duration * 0.2)) * 20)  # 0-20% in first 20%
            elif elapsed < estimated_duration * 0.6:
                remaining_time = elapsed - (estimated_duration * 0.2)
                remaining_duration = estimated_duration * 0.4
                progress = 20 + int((remaining_time / remaining_duration) * 60)  # 20-80% in next 40%
            else:
                remaining_time = elapsed - (estimated_duration * 0.6)
                remaining_duration = estimated_duration * 0.4
                progress = 80 + int((remaining_time / remaining_duration) * 15)  # 80-95% in last 40%
            
            return {
                'progress': min(95, max(0, progress)),
                'progress_message': 'Processing your request...',
                'estimated_completion': created_at + estimated_duration
            }
            
        except:
            return {
                'progress': 50,
                'progress_message': 'Processing...',
                'estimated_completion': time.time() + 60
            }

    def create_cache_entry(self, request_hash, payload, status='processing', estimated_duration=60):
        """Create new cache entry for request"""
        if not self.cache_table:
            print("⚠️ Cache table not available")
            return False
            
        try:
            current_time = time.time()
            ttl = int(current_time + (24 * 60 * 60))  # 24 hour TTL
            
            cache_item = {
                'request_hash': request_hash,
                'status': status,
                'created_at': current_time,
                'updated_at': current_time,
                'estimated_duration': estimated_duration,
                'original_payload': json.dumps(payload, default=str),
                'user_id': payload.get('user_id', ''),
                'agent_type': payload.get('agent_type', ''),
                'message_preview': payload.get('message', '')[:200],
                'ttl': ttl
            }
            
            self.cache_table.put_item(Item=cache_item)
            
            # Create initial progress entry
            if status == 'processing':
                self._update_progress(request_hash, 0, 'Request queued for processing')
            
            print(f"💾 Created cache entry {request_hash[:8]} with status: {status}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating cache entry: {e}")
            return False

    def _update_cache_status(self, request_hash, status, error=None, processing_time=None):
        """Update cache entry status"""
        if not self.cache_table:
            return
            
        try:
            update_expression = "SET #status = :status, updated_at = :updated_at"
            expression_values = {
                ':status': status,
                ':updated_at': time.time()
            }
            expression_names = {'#status': 'status'}
            
            if error:
                update_expression += ", #error = :error"
                expression_values[':error'] = str(error)
                expression_names['#error'] = 'error'
                
            if processing_time:
                update_expression += ", processing_time = :processing_time"
                expression_values[':processing_time'] = processing_time
            
            self.cache_table.update_item(
                Key={'request_hash': request_hash},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ExpressionAttributeNames=expression_names
            )
            
            print(f"📝 Updated cache {request_hash[:8]} status to: {status}")
            
        except Exception as e:
            print(f"⚠️ Error updating cache status: {e}")

    def save_response(self, request_hash, result):
        """Save full response data"""
        if not self.response_table:
            print("⚠️ Response table not available")
            return
            
        try:
            current_time = time.time()
            ttl = int(current_time + (24 * 60 * 60))  # 24 hour TTL
            
            response_item = {
                'request_hash': request_hash,
                'response': str(result.get('response', ''))[:10000],  # Limit response size
                'agent_type': result.get('agent_type', ''),
                'processing_time': result.get('processing_time', 0),
                'image_urls': json.dumps(result.get('image_urls', []), default=str),
               'audio_urls': json.dumps(result.get('audio_urls', []), default=str),
               'files_created': json.dumps(result.get('files_created', []), default=str),
               'tools_used': json.dumps(result.get('strands_tools_available', []), default=str),
               'metadata': json.dumps({
                   'timestamp': result.get('timestamp', ''),
                   'success': True,
                   'cached_at': current_time
               }, default=str),
               'created_at': current_time,
               'ttl': ttl
           }

            self.response_table.put_item(Item=response_item)
           
           # Update main cache status
            self._update_cache_status(request_hash, 'completed', processing_time=result.get('processing_time', 0))
           
           # Final progress update
            self._update_progress(request_hash, 100, 'Request completed successfully')
           
            print(f"💾 Saved response for {request_hash[:8]}")
           
        except Exception as e:
           print(f"❌ Error saving response: {e}")
           self._update_cache_status(request_hash, 'failed', error=str(e))
        
    def save_error(self, request_hash, error):
       """Save error information"""
       try:
           error_msg = str(error)
           self._update_cache_status(request_hash, 'failed', error=error_msg)
           self._update_progress(request_hash, 0, f'Request failed: {error_msg[:100]}')
           
           print(f"❌ Saved error for {request_hash[:8]}: {error_msg[:100]}")
           
       except Exception as e:
           print(f"❌ Error saving error info: {e}")
    
    def _update_progress(self, request_hash, progress, message, phase=None):
       """Update progress information"""
       if not self.progress_table:
           return
           
       try:
           current_time = time.time()
           ttl = int(current_time + (6 * 60 * 60))  # 6 hour TTL for progress
           
           update_expression = "SET progress = :progress, progress_message = :message, updated_at = :updated_at, ttl = :ttl"
           expression_values = {
               ':progress': int(progress),
               ':message': str(message),
               ':updated_at': current_time,
               ':ttl': ttl
           }
           
           if phase:
               update_expression += ", current_phase = :phase"
               expression_values[':phase'] = phase
           
           self.progress_table.update_item(
               Key={'request_hash': request_hash},
               UpdateExpression=update_expression,
               ExpressionAttributeValues=expression_values,
               ReturnValues="NONE"
           )
           
           print(f"📊 Progress {request_hash[:8]}: {progress}% - {message}")
           
       except Exception as e:
           print(f"⚠️ Error updating progress: {e}")

    def is_request_active(self, request_hash):
       """Check if request is currently being processed"""
       return request_hash in self.active_requests

    def mark_request_active(self, request_hash, thread_id):
       """Mark request as actively processing"""
       self.active_requests[request_hash] = {
           'thread_id': thread_id,
           'started_at': time.time(),
           'status': 'active'
       }

    def mark_request_inactive(self, request_hash):
       """Remove request from active processing"""
       if request_hash in self.active_requests:
           del self.active_requests[request_hash]

class EnhancedMultiAgentChatSystem:
   def __init__(self):
       """Initialize enhanced multi-agent chat system with advanced caching"""
       self.session = boto3.Session(region_name='us-east-1')
       self.s3_client = boto3.client('s3', region_name='us-east-1')
       self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
       
       # Initialize enhanced cache manager
       self.cache_manager = EnhancedCacheManager(self.dynamodb)
       
       # Chat history table
       self.chat_table_name = 'workflow-agent-chat'
       self._ensure_chat_table_exists()
       
       # Agent cache
       self.agents = {}
       
       # ADD MARKETING CAMPAIGN MANAGER INITIALIZATION
       self.marketing_manager = MarketingCampaignManager(Environment.PRODUCTION)
       
       # Strands tools mapping
       self.strands_tools = {}
       
       # Add tools that were successfully imported
       if use_aws:
           self.strands_tools["use_aws"] = use_aws
       if speak:
           self.strands_tools["speak"] = speak
       if generate_image:
           self.strands_tools["generate_image"] = generate_image
       if nova_reels:
           self.strands_tools["nova_reels"] = nova_reels
       if image_reader:
           self.strands_tools["image_reader"] = image_reader
       if retrieve:
           self.strands_tools["retrieve"] = retrieve
       if memory:
           self.strands_tools["memory"] = memory
       if file_read:
           self.strands_tools["file_read"] = file_read
       if file_write:
           self.strands_tools["file_write"] = file_write
       if http_request:
           self.strands_tools["http_request"] = http_request
       if python_repl:
           self.strands_tools["python_repl"] = python_repl
       if calculator:
           self.strands_tools["calculator"] = calculator
       if current_time:
           self.strands_tools["current_time"] = current_time
       if workflow:
           self.strands_tools["workflow"] = workflow
       if use_llm:
           self.strands_tools["use_llm"] = use_llm
       
       print(f"🔧 Initialized with {len(self.strands_tools)} Strands tools: {list(self.strands_tools.keys())}")
       print(f"🎯 Marketing Campaign Manager integrated successfully")
       
   def _ensure_chat_table_exists(self):
       """Ensure chat history table exists"""
       try:
           self.chat_table = self.dynamodb.Table(self.chat_table_name)
           self.chat_table.load()
           print(f"✅ Using existing chat table: {self.chat_table_name}")
       except ClientError as e:
           if e.response['Error']['Code'] == 'ResourceNotFoundException':
               print(f"🔧 Creating chat table: {self.chat_table_name}")
               self.chat_table = self.dynamodb.create_table(
                   TableName=self.chat_table_name,
                   KeySchema=[
                       {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                       {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                   ],
                   AttributeDefinitions=[
                       {'AttributeName': 'user_id', 'AttributeType': 'S'},
                       {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                   ],
                   BillingMode='PAY_PER_REQUEST'
               )
               self.chat_table.wait_until_exists()
               print("✅ Chat table created successfully")
           else:
               self.chat_table = None

   def _is_long_running_request(self, payload):
       """UPDATED: Enhanced detection for requests that need background processing"""
       message = payload.get('message', '').lower()
       agent_type = payload.get('agent_type', '')
       context = payload.get('context', '').lower()
       
       # Length-based indicators
       message_length = len(message)
       context_length = len(context)
       
       # UPDATED: Add marketing campaign indicators
       marketing_indicators = [
           'marketing campaign' in message and ('generate' in message or 'create' in message),
           'customer list' in message or 'customer data' in message,
           'personalized message' in message and ('multiple' in message or 'batch' in message),
           'campaign generation' in message,
           'cold outreach' in message and ('customers' in message or 'prospects' in message),
           agent_type == 'marketing_campaign_agent' and message_length > 100,
           'csv' in context or 'spreadsheet' in context,
           context_length > 300  # Large customer data context
       ]
       
       # Content-based indicators for long-running operations
       long_running_indicators = [
           # Image/Video generation
           'generate image' in message and ('detailed' in message or 'complex' in message),
           'create image' in message and message_length > 100,
           'video' in message or 'nova_reels' in message,
           
           # Analysis and processing
           'comprehensive analysis' in message,
           'detailed report' in message,
           'analyze' in message and ('data' in message or 'file' in message),
           
           # Multiple operations
           message.count('and') >= 3,  # Multiple tasks
           message.count(',') >= 5,    # Complex instructions
           'workflow' in message and ('execute' in message or 'multiple' in message),
           
           # Time-intensive operations
           'convert' in message and 'speech' in message and message_length > 200,
           'transcribe' in message or 'translation' in message,
           'optimization' in message or 'training' in message,
           
           # Large content processing
           'process' in message and ('document' in message or 'file' in message),
           context_length > 500,  # Large context
           
           # Agent-specific operations
           agent_type in ['multimodal_agent', 'workflow_agent', 'analytics', 'marketing_campaign_agent'] and message_length > 150,
       ]
       
       # File upload indicator (always considered potentially long-running)
       has_file_upload = payload.get('file_upload') is not None
       
       # Complex agent types that typically take longer
       complex_agent_types = [
           'multimodal_agent',
           'workflow_agent', 
           'analytics',
           'aws_ai_agent',
           'marketing_campaign_agent'  # ADDED
       ]
       
       # Scoring system for long-running detection
       score = 0
       score += min(4, message_length // 150)  # 1 point per 150 chars, max 4
       score += min(3, context_length // 200)   # 1 point per 200 chars, max 3
       score += sum(long_running_indicators)    # 1 point per indicator
       score += sum(marketing_indicators)       # ADDED: Marketing indicators
       score += 3 if agent_type in complex_agent_types else 0
       score += 4 if has_file_upload else 0     # File uploads often take longer
       
       # Keywords that strongly suggest long processing
       if any(word in message for word in ['comprehensive', 'detailed analysis', 'multiple images', 'batch process', 'campaign generation']):
           score += 3
           
       print(f"📊 Long-running request scoring - Message: {message_length}chars, Agent: {agent_type}, Score: {score}/20")
       
       return score >= 6  # Threshold for background processing

   def _estimate_processing_duration(self, payload):
       """UPDATED: Estimate processing duration based on request complexity"""
       base_duration = 30  # Base 30 seconds
       
       message = payload.get('message', '').lower()
       agent_type = payload.get('agent_type', '')
       
       # Duration modifiers based on content
       duration_modifiers = {
           'image generation': 45,
           'video creation': 120,
           'speech synthesis': 30,
           'file processing': 60,
           'comprehensive analysis': 90,
           'workflow execution': 75,
           'multiple operations': 60,
           'marketing campaign': 120,  # ADDED
           'campaign generation': 90,  # ADDED
           'customer analysis': 60     # ADDED
       }
       
       for operation, additional_time in duration_modifiers.items():
           if operation.replace(' ', '') in message.replace(' ', ''):
               base_duration += additional_time
               
       # Agent-specific duration modifiers
       agent_duration_map = {
           'multimodal_agent': 60,
           'workflow_agent': 45,
           'analytics': 40,
           'aws_ai_agent': 35,
           'image_generator': 50,
           'marketing_campaign_agent': 90  # ADDED
       }
       
       if agent_type in agent_duration_map:
           base_duration += agent_duration_map[agent_type]
           
       # File upload adds processing time
       if payload.get('file_upload'):
           base_duration += 45
           
       # Context complexity (especially for customer data)
       context_length = len(payload.get('context', ''))
       if context_length > 500:
           base_duration += min(60, context_length // 100)
           
       return min(base_duration, 300)  # Cap at 5 minutes

   def process_chat_request(self, payload):
       """Main method to process chat requests with enhanced caching"""
       request_hash = self.cache_manager._generate_request_hash(payload)
       
       print(f"🔄 Processing request {request_hash[:8]} for user {payload.get('user_id', 'unknown')}")
       
       # Check cache first
       cached_request = self.cache_manager.get_cached_request(request_hash)
       
       if cached_request:
           status = cached_request['status']
           
           if status == 'completed':
               print(f"✅ Returning completed cached result for {request_hash[:8]}")
               return {
                   'success': True,
                   'status': 'completed',
                   'request_hash': request_hash,
                   'from_cache': True,
                   **{k: v for k, v in cached_request.items() 
                      if k not in ['status', 'request_hash']}
               }
               
           elif status == 'processing':
               print(f"⏳ Request {request_hash[:8]} is already processing")
               return {
                   'success': True,
                   'status': 'processing',
                   'request_hash': request_hash,
                   'progress': cached_request.get('progress', 0),
                   'message': cached_request.get('progress_message', 'Processing your request...'),
                   'estimated_completion': cached_request.get('estimated_completion', time.time() + 60),
                   'poll_url': f"/poll/{request_hash}",
                   'poll_interval': 5,
                   'from_cache': True
               }
               
           elif status == 'failed':
               print(f"❌ Request {request_hash[:8]} previously failed")
               # For failed requests, we might want to retry automatically or return the cached error
               if time.time() - cached_request.get('updated_at', 0) > 300:  # 5 minutes
                   print(f"🔄 Retrying previously failed request {request_hash[:8]}")
                   # Continue to process as new request
               else:
                   return {
                       'success': False,
                       'status': 'failed',
                       'request_hash': request_hash,
                       'error': cached_request.get('error', 'Unknown error occurred'),
                       'from_cache': True
                   }
       
       # Determine if this is a long-running request
       is_long_running = self._is_long_running_request(payload)
       estimated_duration = self._estimate_processing_duration(payload)
       
       if is_long_running:
           print(f"🔄 Long-running request detected, processing in background: {request_hash[:8]}")
           
           # Create cache entry for background processing
           self.cache_manager.create_cache_entry(
               request_hash, 
               payload, 
               status='processing', 
               estimated_duration=estimated_duration
           )
           
           # Start background processing
           thread = Thread(target=self._process_in_background, args=(payload, request_hash))
           thread.daemon = True
           thread.start()
           
           # Mark as actively processing
           self.cache_manager.mark_request_active(request_hash, thread.ident)
           
           return {
               'success': True,
               'status': 'processing',
               'request_hash': request_hash,
               'message': f'Long-running request detected. Processing in background.',
               'progress': 0,
               'estimated_duration': estimated_duration,
               'estimated_completion': time.time() + estimated_duration,
               'poll_url': f"/poll/{request_hash}",
               'poll_interval': 5,
               'from_cache': False
           }
       
       # Process short requests immediately
       print(f"⚡ Processing short request immediately: {request_hash[:8]}")
       
       # Create cache entry for immediate processing
       self.cache_manager.create_cache_entry(request_hash, payload, status='processing', estimated_duration=30)
       
       try:
           result = self._process_message_immediately(payload)
           
           if result['success']:
               # Save successful result
               self.cache_manager.save_response(request_hash, result)
               
               return {
                   'success': True,
                   'status': 'completed',
                   'request_hash': request_hash,
                   'from_cache': False,
                   **{k: v for k, v in result.items() if k != 'success'}
               }
           else:
               # Save error
               self.cache_manager.save_error(request_hash, result.get('error'))
               
               return {
                   'success': False,
                   'status': 'failed',
                   'request_hash': request_hash,
                   'error': result.get('error'),
                   'from_cache': False
               }
               
       except Exception as e:
           print(f"❌ Error processing immediate request {request_hash[:8]}: {e}")
           self.cache_manager.save_error(request_hash, str(e))
           
           return {
               'success': False,
               'status': 'failed',
               'request_hash': request_hash,
               'error': str(e),
               'from_cache': False
           }

   def _process_in_background(self, payload, request_hash):
       """Enhanced background processing with comprehensive progress tracking"""
       try:
           print(f"🔄 Starting background processing for {request_hash[:8]}")
           
           # Phase 1: Initialization (0-10%)
           self.cache_manager._update_progress(
               request_hash, 5, "Initializing agent and loading tools...", "initialization"
           )
           
           # Get user configuration and determine agent
           user_id = payload.get('user_id')
           workflow_data = self.fetch_workflow_config(user_id)
           agent_type = payload.get('agent_type') or self.determine_agent_type(payload.get('message'), workflow_data)
           
           # Phase 2: Agent Setup (10-20%)
           self.cache_manager._update_progress(
               request_hash, 15, f"Setting up {agent_type} agent with required tools...", "agent_setup"
           )
           
           agent = self.get_or_create_agent(user_id, agent_type)
           
           # Phase 3: Pre-processing (20-30%)
           self.cache_manager._update_progress(
               request_hash, 25, "Processing request parameters and context...", "preprocessing"
           )
           
           # Handle file upload if present
           file_url = None
           if payload.get('file_upload'):
               self.cache_manager._update_progress(
                   request_hash, 30, "Processing uploaded file...", "file_processing"
               )
               file_url = self.process_file_upload(user_id, payload['file_upload'])
           
           # Phase 4: Main Processing (30-85%)
           self.cache_manager._update_progress(
               request_hash, 35, "Processing your request with AI agent...", "main_processing"
           )
           
           # Simulate incremental progress during processing
           progress_thread = Thread(
               target=self._simulate_detailed_progress, 
               args=(request_hash, 35, 85, 10)  # From 35% to 85% over ~10 intervals
           )
           progress_thread.daemon = True
           progress_thread.start()
           
           # Actually process the message
           start_time = time.time()
           result = self._process_message_with_agent(agent, payload, file_url)
           processing_time = time.time() - start_time
           
           # Phase 5: Post-processing (85-95%)
           self.cache_manager._update_progress(
               request_hash, 90, "Finalizing response and saving results...", "postprocessing"
           )
           
           if result['success']:
               result['processing_time'] = processing_time
               result['request_hash'] = request_hash
               result['background_processed'] = True
               
               # Phase 6: Completion (95-100%)
               self.cache_manager._update_progress(
                   request_hash, 95, "Saving response to cache...", "completion"
               )
               
               # Save successful result
               self.cache_manager.save_response(request_hash, result)
               
               print(f"✅ Background processing completed successfully for {request_hash[:8]}")
               
           else:
               # Save error result
               self.cache_manager.save_error(request_hash, result.get('error', 'Unknown error'))
               print(f"❌ Background processing failed for {request_hash[:8]}: {result.get('error')}")
               
       except Exception as e:
           print(f"❌ Background processing exception for {request_hash[:8]}: {e}")
           import traceback
           traceback.print_exc()
           
           self.cache_manager.save_error(request_hash, str(e))
           
       finally:
           # Clean up
           self.cache_manager.mark_request_inactive(request_hash)

   def _simulate_detailed_progress(self, request_hash, start_progress, end_progress, steps):
       """Simulate detailed progress updates during main processing"""
       progress_messages = [
           "Analyzing request parameters...",
           "Loading AI models and tools...", 
           "Processing with neural networks...",
           "Generating content...",
           "Applying transformations...",
           "Running quality checks...",
           "Optimizing output...",
           "Preparing final response...",
           "Validating results...",
           "Almost complete..."
       ]
       
       step_size = (end_progress - start_progress) / steps
       step_duration = 8  # seconds per step
       
       for i in range(steps):
           time.sleep(step_duration)
           
           current_progress = int(start_progress + (i * step_size))
           message_idx = min(i, len(progress_messages) - 1)
           
           try:
               self.cache_manager._update_progress(
                   request_hash,
                   current_progress,
                   progress_messages[message_idx],
                   "main_processing"
               )
           except:
               break  # Stop if progress update fails

   def poll_request_status(self, request_hash):
       """Poll the status of a processing request"""
       print(f"🔍 Polling status for request {request_hash[:8]}")
       
       cached_request = self.cache_manager.get_cached_request(request_hash)
       
       if not cached_request:
           return {
               'success': False,
               'status': 'not_found',
               'error': 'Request not found, expired, or never existed',
               'request_hash': request_hash
           }
       
       status = cached_request['status']
       
       if status == 'completed':
           print(f"✅ Request {request_hash[:8]} completed")
           return {
               'success': True,
               'status': 'completed',
               'request_hash': request_hash,
               **{k: v for k, v in cached_request.items() 
                  if k not in ['status', 'request_hash']}
           }
           
       elif status == 'processing':
           elapsed_time = int(time.time() - cached_request.get('created_at', time.time()))
           estimated_remaining = max(0, cached_request.get('estimated_duration', 60) - elapsed_time)
           
           return {
               'success': True,
               'status': 'processing',
               'request_hash': request_hash,
               'progress': cached_request.get('progress', 0),
               'message': cached_request.get('progress_message', 'Processing your request...'),
               'current_phase': cached_request.get('current_phase', 'processing'),
               'elapsed_time': elapsed_time,
               'estimated_remaining': estimated_remaining,
               'estimated_completion': cached_request.get('estimated_completion', time.time() + 60),
               'poll_interval': 5
           }
           
       else:  # failed
           return {
               'success': False,
               'status': 'failed',
               'request_hash': request_hash,
               'error': cached_request.get('error', 'Unknown error occurred'),
               'elapsed_time': int(time.time() - cached_request.get('created_at', time.time()))
           }

   def _process_message_immediately(self, payload):
       """Process message immediately for short requests"""
       try:
           user_id = payload.get('user_id')
           
           # Get workflow config and agent
           workflow_data = self.fetch_workflow_config(user_id)
           agent_type = payload.get('agent_type') or self.determine_agent_type(payload.get('message'), workflow_data)
           agent = self.get_or_create_agent(user_id, agent_type)
           
           # Handle file upload if present
           file_url = None
           if payload.get('file_upload'):
               file_url = self.process_file_upload(user_id, payload['file_upload'])
           
           # Process with agent
           return self._process_message_with_agent(agent, payload, file_url)
           
       except Exception as e:
           print(f"❌ Error in immediate processing: {e}")
           return {'success': False, 'error': str(e)}

   def _process_message_with_agent(self, agent, payload, file_url=None):
       """Process message with the given agent"""
       try:
           user_id = payload.get('user_id')
           message = payload.get('message')
           context = payload.get('context')
           
           start_time = time.time()
           
           # Enhance message with context and file
           enhanced_message = self._enhance_message(user_id, message, context, file_url)
           
           # Process with agent
           response = agent(enhanced_message)
           response_text = self._extract_response_text(response)
           
           # Extract various content types from response
           image_urls = self._extract_content_urls(response, 'image')
           audio_urls = self._extract_content_urls(response, 'audio')
           files_created = self._extract_content_urls(response, 'file')
           
           # Save interaction to chat history
           self.save_chat_message(user_id, message, response_text, payload.get('agent_type', 'unknown'))
           
           processing_time = time.time() - start_time
           
           result = {
               'response': response_text,
               'agent_type': payload.get('agent_type', 'unknown'),
               'processing_time': processing_time,
               'strands_tools_available': list(self.strands_tools.keys()),
               'timestamp': datetime.now().isoformat()
           }
           
           # Add content URLs if present
           if image_urls:
               result['image_urls'] = image_urls
           if audio_urls:
               result['audio_urls'] = audio_urls
           if files_created:
               result['files_created'] = files_created
           
           return {'success': True, **result}
           
       except Exception as e:
           print(f"❌ Error processing message with agent: {e}")
           return {'success': False, 'error': str(e)}

   def _enhance_message(self, user_id, message, context=None, file_url=None):
       """Enhance message with context, history, and file information"""
       enhanced_parts = []
       
       # Add context if provided
       if context:
           enhanced_parts.append(f"Context: {context}")
       
       # Add file information if provided
       if file_url:
           enhanced_parts.append(f"Uploaded file URL: {file_url}")
       
       # Add recent chat history for continuity
       chat_history = self.get_chat_history(user_id, limit=3)
       if chat_history:
           history_context = "\n".join([
               f"Previous: {item['message'][:100]} -> {str(item['response'])[:100]}..."
               for item in reversed(chat_history)
           ])
           enhanced_parts.append(f"Recent conversation history:\n{history_context}")
       
       # Add current message
       enhanced_parts.append(f"Current request: {message}")
       
       return "\n\n".join(enhanced_parts)

   def fetch_workflow_config(self, user_id):
       """Fetch workflow configuration from S3"""
       try:
           bucket_name = 'qubitz-customer-prod'
           
           # Try user-specific config first
           try:
               key = f'{user_id}/workflow.json'
               response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
               workflow_data = json.loads(response['Body'].read().decode('utf-8'))
               print(f"✅ User-specific workflow config loaded for {user_id}")
               return workflow_data
           except:
               # Try project-specific config
               try:
                   objects = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f'{user_id}/')
                   if 'Contents' in objects and objects['Contents']:
                       for obj in objects['Contents']:
                           if obj['Key'].endswith('.json'):
                               key = obj['Key']
                               response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                               workflow_data = json.loads(response['Body'].read().decode('utf-8'))
                               print(f"✅ Project workflow config loaded for {user_id}: {key}")
                               return workflow_data
               except:
                   pass
           
           print(f"⚠️ No workflow config found for {user_id}")
           return None
           
       except Exception as e:
           print(f"⚠️ Error fetching workflow config: {e}")
           return None

   def determine_agent_type(self, message, workflow_data=None):
       """UPDATED: Determine which agent type to use based on message content"""
       message_lower = message.lower()
       
       # UPDATED: Add marketing campaign detection (NEW)
       if any(word in message_lower for word in [
           'marketing campaign', 'cold outreach', 'lead generation', 'customer outreach',
           'marketing message', 'campaign generation', 'personalized message', 'b2b outreach',
           'sales campaign', 'customer campaign', 'marketing automation', 'cold messaging',
           'prospect outreach', 'email campaign'
       ]):
           return 'marketing_campaign_agent'
       
       # Enhanced agent type detection
       elif any(word in message_lower for word in ['generate image', 'create image', 'make picture', 'draw', 'illustrate']):
           return 'image_generator'
       elif any(word in message_lower for word in ['speak', 'voice', 'audio', 'tts', 'text to speech', 'polly']):
           return 'aws_ai_agent'
       elif any(word in message_lower for word in ['video', 'nova_reels', 'reel', 'animation']):
           return 'multimodal_agent'
       elif any(word in message_lower for word in ['sentiment', 'analyze text', 'comprehend', 'translate', 'aws']):
           return 'aws_ai_agent'
       elif any(word in message_lower for word in ['workflow', 'orchestrate', 'process', 'coordinate', 'chain']):
           return 'workflow_agent'
       elif any(word in message_lower for word in ['validate', 'check', 'verify', 'review', 'compliance']):
           return 'validator'
       elif any(word in message_lower for word in ['analyze', 'analytics', 'report', 'metrics', 'calculate', 'math']):
           return 'analytics'
       elif any(word in message_lower for word in ['comprehensive', 'detailed', 'multiple', 'complex']):
           return 'multimodal_agent'
       else:
           return 'ai_chat_agent'
    
   def create_agent_from_workflow(self, workflow_data, agent_type):
       """UPDATED: Create specialized agent based on workflow configuration"""
       system_prompts = {
           # UPDATED: Add marketing campaign agent (NEW)
           'marketing_campaign_agent': f"""
You are an expert B2B marketing campaign specialist integrated with advanced AI tools.

MARKETING CAPABILITIES:
- Generate personalized cold outreach messages
- Create industry-specific messaging
- Develop customer segmentation strategies  
- Analyze customer data for lead scoring
- Create multi-channel campaign content

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

MARKETING PRINCIPLES:
- Always create truthful, non-misleading content
- Personalize based on industry and role
- Focus on value proposition, not hard selling
- Ask thoughtful questions to engage prospects
- Respect professional communication standards

When users request marketing campaigns:
1. Parse customer data from context (name, company, position, email)
2. Analyze industry and seniority level for each customer
3. Generate personalized messages using appropriate tone
4. Create both quick-connect (50-75 words) and detailed (175-200 words) versions
5. Ensure messages are truthful and professional

CRITICAL: Never invent statistics or fake achievements. Use phrases like "designed to help" instead of definitive claims.

Be professional, strategic, and results-focused while maintaining ethical standards.
           """,
           
           'ai_chat_agent': f"""
You are an intelligent customer support chat agent with access to advanced tools and AWS services.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Provide helpful, professional responses and use appropriate tools when beneficial.
Be conversational and ask clarifying questions when needed.
           """,
           
           'ai_text_generator': f"""
You are an AI text generation specialist capable of creating high-quality content.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Generate professional, well-structured content based on user requirements.
Use tools like file_write to save content when requested.
           """,
           
           'validator': f"""
You are a validation specialist with advanced analysis capabilities.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Validate data, check compliance, and provide comprehensive feedback.
Use python_repl for computational validation when needed.
           """,
           
           'analytics': f"""
You are an analytics specialist with computational and AI-powered insights.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Perform data analysis, create reports, and provide actionable insights.
Use calculator and python_repl for mathematical operations.
           """,
           
           'image_generator': f"""
You are an AI image generation and analysis specialist.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Create images with generate_image, analyze with image_reader, and explain your process.
Provide detailed descriptions of generated content.
           """,
           
           'aws_ai_agent': f"""
You are an AWS AI services coordinator with comprehensive cloud AI capabilities.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Use AWS services via use_aws tool for Polly (TTS), Comprehend (sentiment), 
Textract (documents), Rekognition (images), and Translate.
           """,
           
           'multimodal_agent': f"""
You are a multimodal AI agent capable of handling text, images, video, and audio.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Combine multiple tools to create rich, multimodal responses.
Generate images, videos, audio, and text as needed.
           """,
           
           'workflow_agent': f"""
You are a workflow orchestration agent managing complex processes.

AVAILABLE STRANDS TOOLS: {list(self.strands_tools.keys())}

Orchestrate complex workflows using multiple tools in sequence.
Break down complex tasks into manageable steps.
           """
       }
       
       system_prompt = system_prompts.get(agent_type, system_prompts['ai_chat_agent'])
       
       # UPDATED: Use environment-specific model selection
       model_config = self.marketing_manager.current_model if agent_type == 'marketing_campaign_agent' else None
       
       if model_config:
           # Use marketing-specific model configuration
           bedrock_model = BedrockModel(
               model_id=model_config.model_id,
               boto_session=self.session,
               max_tokens=4000,
               params={"temperature": 0.3, "top_p": 0.9}
           )
       else:
           # Use default model for non-marketing agents
           bedrock_model = BedrockModel(
               model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
               boto_session=self.session,
               max_tokens=4000,
               params={"temperature": 0.3, "top_p": 0.9}
           )
       
       # Get tools for this agent
       agent_tools = self._get_agent_tools(agent_type)
       
       return Agent(
           model=bedrock_model,
           conversation_manager=SlidingWindowConversationManager(window_size=5),
           system_prompt=system_prompt,
           tools=agent_tools
       )

   def _get_agent_tools(self, agent_type):
       """UPDATED: Get tools for specific agent type"""
       agent_tool_mappings = {
           # UPDATED: Add marketing campaign agent tools (NEW)
           'marketing_campaign_agent': [
               "use_aws",          # For AWS AI services (Comprehend, Translate, etc.)  
               "http_request",     # For web research and API calls
               "file_read",        # For processing customer CSV files
               "file_write",       # For saving campaign results
               "python_repl",      # For data analysis and processing
               "calculator",       # For campaign metrics
               "current_time",     # For scheduling
               "memory"            # For campaign context
           ],
           
           'ai_chat_agent': ["use_aws", "speak", "memory", "calculator", "current_time"],
           'ai_text_generator': ["use_aws", "speak", "file_write", "file_read"],
           'validator': ["use_aws", "python_repl", "file_read", "calculator"],
           'analytics': ["python_repl", "calculator", "use_aws", "file_read", "file_write"],
           'image_generator': ["generate_image", "image_reader", "use_aws"],
           'aws_ai_agent': ["use_aws", "speak", "image_reader", "memory"],
           'multimodal_agent': ["generate_image", "nova_reels", "image_reader", "speak", "use_aws"],
           'workflow_agent': ["workflow", "use_llm", "use_aws", "python_repl"],
       }
       
       tool_names = agent_tool_mappings.get(agent_type, ["use_aws", "calculator"])
       
       tools = []
       for tool_name in tool_names:
           if tool_name in self.strands_tools:
               tools.append(self.strands_tools[tool_name])
       
       print(f"🤖 Agent {agent_type} initialized with {len(tools)} tools")
       return tools

   def get_or_create_agent(self, user_id, agent_type):
       """Get or create agent for user and type"""
       agent_key = f"{user_id}_{agent_type}"
       
       if agent_key not in self.agents:
           print(f"🤖 Creating new agent: {agent_type} for user {user_id}")
           workflow_data = self.fetch_workflow_config(user_id)
           self.agents[agent_key] = self.create_agent_from_workflow(workflow_data, agent_type)
           
       return self.agents[agent_key]

   def save_chat_message(self, user_id, message, response, agent_type):
       """Save chat interaction to DynamoDB"""
       if not self.chat_table:
           return
           
       try:
           timestamp = int(time.time() * 1000)
           
           self.chat_table.put_item(Item={
               'user_id': user_id,
               'timestamp': timestamp,
               'message': message[:1000],  # Truncate if too long
               'response': str(response)[:1000],  # Truncate if too long
               'agent_type': agent_type,
               'created_at': datetime.now().isoformat()
           })
           
           print(f"💾 Saved chat interaction for {user_id}")
       except Exception as e:
           print(f"⚠️ Error saving chat message: {e}")

   def get_chat_history(self, user_id, limit=10):
       """Get recent chat history for user"""
       if not self.chat_table:
           return []
           
       try:
           response = self.chat_table.query(
               KeyConditionExpression='user_id = :user_id',
               ExpressionAttributeValues={':user_id': user_id},
               ScanIndexForward=False,
               Limit=limit
           )
           
           return response.get('Items', [])
       except Exception as e:
           print(f"⚠️ Error getting chat history: {e}")
           return []

   def process_file_upload(self, user_id, file_upload):
       """Process file upload and store in S3"""
       try:
           if isinstance(file_upload, dict):
               file_name = file_upload.get('filename')
               file_content = base64.b64decode(file_upload.get('content'))
               content_type = file_upload.get('contentType', 'application/octet-stream')
           else:
               print("⚠️ Invalid file_upload format")
               return None
           
           file_id = str(uuid.uuid4())
           bucket_name = 'qubitz-customer-prod'
           key = f'{user_id}/files/{file_id}_{file_name}'
           
           self.s3_client.put_object(
               Body=file_content,
               Bucket=bucket_name,
               Key=key,
               ContentType=content_type
           )
           
           file_url = self.s3_client.generate_presigned_url(
               'get_object',
               Params={'Bucket': bucket_name, 'Key': key},
               ExpiresIn=3600
           )
           
           print(f"📁 File uploaded: {key}")
           return file_url
           
       except Exception as e:
           print(f"❌ Error processing file upload: {e}")
           return None

   def _extract_response_text(self, response):
       """Extract text from agent response"""
       try:
           if hasattr(response, 'content'):
               if isinstance(response.content, list):
                   text_parts = []
                   for item in response.content:
                       if isinstance(item, dict):
                           if 'text' in item:
                               text_parts.append(item['text'])
                           elif 'content' in item:
                               text_parts.append(str(item['content']))
                       else:
                           text_parts.append(str(item))
                   return ' '.join(text_parts)
               else:
                   return str(response.content)
           else:
               return str(response)
       except Exception as e:
           print(f"⚠️ Error extracting response: {e}")
           return str(response)
   
   def _extract_content_urls(self, response, content_type):
       """Extract URLs from response for specific content type"""
       try:
           urls = []
           if hasattr(response, 'content') and isinstance(response.content, list):
               for item in response.content:
                   if isinstance(item, dict):
                       if f'{content_type}_url' in item:
                           urls.append(item[f'{content_type}_url'])
                       elif content_type in item and isinstance(item[content_type], dict):
                           if 'url' in item[content_type]:
                               urls.append(item[content_type]['url'])
                           elif 'source' in item[content_type]:
                               urls.append(item[content_type]['source'])
           return urls if urls else None
       except Exception as e:
           print(f"⚠️ Error extracting {content_type} URLs: {e}")
           return None

# Global chat system instance
chat_system = None

def lambda_handler(event, context):
   """UPDATED: Enhanced Lambda handler with comprehensive caching and polling support"""
   global chat_system
   
   try:
       # Initialize system
       if chat_system is None:
           print("🔧 Initializing Enhanced MultiAgent Chat System with Marketing Campaigns...")
           chat_system = EnhancedMultiAgentChatSystem()
           print("✅ Enhanced MultiAgent Chat System with Marketing Campaigns initialized successfully")
       
       # Parse HTTP method and path
       http_method = event.get('httpMethod', 'POST')
       path = event.get('path', '')
       
       # Handle polling requests (GET /poll/{request_hash})
       if http_method == 'GET' and '/poll/' in path:
           request_hash = path.split('/poll/')[-1]
           print(f"🔍 Polling request for hash: {request_hash[:8]}")
           
           result = chat_system.poll_request_status(request_hash)
           
           return {
               'statusCode': 200,
               'headers': {
                   'Content-Type': 'application/json',
                   'Access-Control-Allow-Origin': '*',
                   'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                   'Access-Control-Allow-Headers': 'Content-Type'
               },
               'body': json.dumps(result, default=str)
           }
       
       # Handle OPTIONS requests for CORS
       if http_method == 'OPTIONS':
           return {
               'statusCode': 200,
               'headers': {
                   'Access-Control-Allow-Origin': '*',
                   'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                   'Access-Control-Allow-Headers': 'Content-Type'
               },
               'body': ''
           }
       
       # Handle chat requests (POST)
       if 'body' in event:
           body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
       else:
           body = event
       
       # Validate required fields
       user_id = body.get('user_id')
       message = body.get('message')
       
       if not user_id or not message:
           return {
               'statusCode': 400,
               'headers': {
                   'Content-Type': 'application/json',
                   'Access-Control-Allow-Origin': '*'
               },
               'body': json.dumps({
                   'success': False,
                   'error': 'Missing required fields: user_id and message',
                   'required_fields': ['user_id', 'message'],
                   'optional_fields': ['context', 'agent_type', 'file_upload'],
                   'supported_agent_types': [
                       'ai_chat_agent', 'ai_text_generator', 'validator',
                       'analytics', 'image_generator', 'aws_ai_agent',
                       'multimodal_agent', 'workflow_agent', 'marketing_campaign_agent'  # ADDED
                   ],
                   'available_tools': list(chat_system.strands_tools.keys()),
                   'marketing_features': {  # ADDED
                       'campaign_generation': 'Generate personalized B2B marketing messages',
                       'industry_targeting': 'Technology, Healthcare, Finance, Retail, Manufacturing, Education, Consulting',
                       'seniority_levels': 'C-Level, Director, Manager, Specialist, Junior',
                       'message_types': 'Quick Connect (50-75 words), Value Proposition (175-200 words)',
                       'truthful_messaging': 'No fake statistics or unverifiable claims'
                   },
                   'caching_info': {
                       'automatic_deduplication': 'Identical requests return cached results',
                       'background_processing': 'Long requests processed asynchronously',
                       'progress_tracking': 'Real-time progress updates available',
                       'cache_duration': '24 hours for completed, 6 hours for failed'
                   },
                   'polling_info': {
                       'poll_endpoint': '/poll/{request_hash}',
                       'poll_interval': '5 seconds recommended',
                       'long_running_threshold': 'Automatically detected'
                   }
               })
           }
       
       print(f"💬 Processing chat request for user: {user_id}")
       print(f"📝 Message preview: {message[:100]}{'...' if len(message) > 100 else ''}")
       print(f"🎯 Agent type: {body.get('agent_type', 'auto-detect')}")
       
       # Process the chat request with enhanced caching
       result = chat_system.process_chat_request(body)
       
       return {
           'statusCode': 200,
           'headers': {
               'Content-Type': 'application/json',
               'Access-Control-Allow-Origin': '*',
               'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
               'Access-Control-Allow-Headers': 'Content-Type'
           },
           'body': json.dumps(result, default=str)
       }
       
   except Exception as e:
       print(f"❌ Lambda error: {e}")
       import traceback
       traceback.print_exc()
       
       return {
           'statusCode': 500,
           'headers': {
               'Content-Type': 'application/json',
               'Access-Control-Allow-Origin': '*'
           },
           'body': json.dumps({
               'success': False,
               'error': str(e),
               'error_type': type(e).__name__,
               'message': 'Internal server error occurred. Please try again or contact support.'
           })
       }

# Helper function for creating marketing campaign requests
def create_marketing_campaign_request(customers: List[dict], offering_details: dict) -> dict:
   """Helper function to create marketing campaign request payload"""
   
   # Convert customer list to context string for the agent
   customer_context = "Customer Data:\n"
   for i, customer in enumerate(customers, 1):
       customer_context += f"{i}. {customer.get('first_name', '')} {customer.get('last_name', '')}"
       if customer.get('company'):
           customer_context += f" at {customer['company']}"
       if customer.get('position'):
           customer_context += f" ({customer['position']})"
       if customer.get('email'):
           customer_context += f" - {customer['email']}"
       customer_context += "\n"
   
   # Create offering context
   offering_context = f"""
Offering Details:
- Name: {offering_details.get('name', 'Not specified')}
- Objective: {offering_details.get('objective', 'Not specified')}
- Benefits: {offering_details.get('benefits', 'Not specified')}
- Target Industry: {offering_details.get('target_industry', 'Not specified')}
- Price Range: {offering_details.get('price_range', 'Not specified')}
- Duration: {offering_details.get('duration', 'Not specified')}
   """
   
   return {
       "user_id": "marketing_campaign_manager",
       "message": f"Generate personalized B2B marketing campaign messages for the following customers. Create both quick-connect (50-75 words) and value proposition (175-200 words) messages for each customer. Ensure messages are truthful, industry-appropriate, and role-specific.",
       "agent_type": "marketing_campaign_agent",
       "context": customer_context + offering_context
   }