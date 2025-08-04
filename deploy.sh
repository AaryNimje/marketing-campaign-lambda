#!/bin/bash

# Marketing Campaign Lambda Deployment Script
set -e

FUNCTION_NAME="arn:aws:lambda:eu-west-2:781364298443:function:markteing-campaign-manager"
REGION="eu-west-2"
PACKAGE_DIR="lambda-package"

echo "🚀 Deploying Marketing Campaign Lambda..."

# Clean and create package directory
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy main function
cp lambda_function.py $PACKAGE_DIR/

# Install Python dependencies (if any new ones)
if [ -f requirements.txt ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt -t $PACKAGE_DIR/
fi

# Remove unnecessary files to reduce package size
find $PACKAGE_DIR -name "*.pyc" -delete
find $PACKAGE_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find $PACKAGE_DIR -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Create deployment zip
cd $PACKAGE_DIR
zip -r ../deployment.zip . -x "*.pyc" "*/__pycache__/*"
cd ..

# Upload to Lambda
echo "📤 Uploading to Lambda..."
aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --zip-file fileb://deployment.zip \
    --region $REGION

# Update function configuration if needed
echo "⚙️ Updating function configuration..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --timeout 900 \
    --memory-size 1024 \
    --region $REGION

echo "✅ Deployment completed successfully!"
echo "🔗 Function ARN: $FUNCTION_NAME"

# Clean up
rm -f deployment.zip
rm -rf $PACKAGE_DIR

echo "🧹 Cleanup completed"