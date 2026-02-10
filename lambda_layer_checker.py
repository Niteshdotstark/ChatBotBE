"""
Lambda Layer Checker - Test if all required packages are available
Deploy this temporarily to your Lambda function to check layers

USAGE:
1. Copy this entire file
2. Paste into your Lambda function (replace existing code)
3. Click Deploy
4. Click Test with any event (e.g., {})
5. Check logs to see which packages are available
6. After checking, replace with your actual code
"""

import json
import sys

def lambda_handler(event, context):
    """Check if all required packages are available"""
    
    print("="*70)
    print("LAMBDA LAYER PACKAGE CHECKER")
    print("="*70)
    print()
    
    results = {
        "python_version": sys.version,
        "packages": {},
        "missing_packages": [],
        "available_packages": []
    }
    
    # List of required packages
    required_packages = {
        "boto3": "AWS SDK",
        "langchain_aws": "LangChain AWS integration",
        "langchain_community": "LangChain community loaders",
        "langchain_text_splitters": "LangChain text splitters",
        "langchain_core": "LangChain core",
        "pypdf": "PDF processing (PyPDF)",
        "docx": "DOCX processing (python-docx)",
    }
    
    print(f"Python Version: {sys.version}")
    print()
    print("Checking required packages...")
    print("-"*70)
    
    # Check each package
    for package_name, description in required_packages.items():
        try:
            if package_name == "pypdf":
                # Try both pypdf and PyPDF2
                try:
                    import pypdf
                    version = getattr(pypdf, '__version__', 'unknown')
                    status = "✅ AVAILABLE"
                    results["packages"][package_name] = {
                        "available": True,
                        "version": version,
                        "description": description
                    }
                    results["available_packages"].append(package_name)
                    print(f"✅ {package_name:30} v{version:15} - {description}")
                except ImportError:
                    try:
                        import PyPDF2
                        version = getattr(PyPDF2, '__version__', 'unknown')
                        status = "✅ AVAILABLE (as PyPDF2)"
                        results["packages"]["PyPDF2"] = {
                            "available": True,
                            "version": version,
                            "description": "PDF processing (PyPDF2)"
                        }
                        results["available_packages"].append("PyPDF2")
                        print(f"✅ PyPDF2 (alternative):30} v{version:15} - PDF processing")
                    except ImportError:
                        status = "❌ MISSING"
                        results["packages"][package_name] = {
                            "available": False,
                            "description": description
                        }
                        results["missing_packages"].append(package_name)
                        print(f"❌ {package_name:30} {'MISSING':15} - {description}")
            
            elif package_name == "docx":
                # python-docx imports as 'docx'
                import docx
                version = getattr(docx, '__version__', 'unknown')
                status = "✅ AVAILABLE"
                results["packages"][package_name] = {
                    "available": True,
                    "version": version,
                    "description": description
                }
                results["available_packages"].append(package_name)
                print(f"✅ {package_name:30} v{version:15} - {description}")
            
            else:
                # Standard import
                module = __import__(package_name)
                version = getattr(module, '__version__', 'unknown')
                status = "✅ AVAILABLE"
                results["packages"][package_name] = {
                    "available": True,
                    "version": version,
                    "description": description
                }
                results["available_packages"].append(package_name)
                print(f"✅ {package_name:30} v{version:15} - {description}")
        
        except ImportError as e:
            status = "❌ MISSING"
            results["packages"][package_name] = {
                "available": False,
                "description": description,
                "error": str(e)
            }
            results["missing_packages"].append(package_name)
            print(f"❌ {package_name:30} {'MISSING':15} - {description}")
            print(f"   Error: {str(e)}")
    
    print()
    print("-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Available packages: {len(results['available_packages'])}/{len(required_packages)}")
    print(f"Missing packages: {len(results['missing_packages'])}")
    print()
    
    if results["missing_packages"]:
        print("❌ MISSING PACKAGES:")
        for pkg in results["missing_packages"]:
            print(f"   - {pkg}")
        print()
        print("ACTION REQUIRED:")
        print("You need to create/update Lambda layer with missing packages")
        print()
        print("To create layer:")
        print("1. On your local machine:")
        print("   mkdir python")
        for pkg in results["missing_packages"]:
            if pkg == "pypdf":
                print(f"   pip install pypdf -t python/")
            elif pkg == "docx":
                print(f"   pip install python-docx -t python/")
            else:
                print(f"   pip install {pkg} -t python/")
        print("   zip -r lambda-layer.zip python/")
        print()
        print("2. Upload to AWS Lambda Layers")
        print("3. Attach layer to your function")
    else:
        print("✅ ALL PACKAGES AVAILABLE!")
        print("You can now deploy your Lambda function code")
    
    print()
    print("="*70)
    
    # Additional checks
    print()
    print("ADDITIONAL CHECKS")
    print("-"*70)
    
    # Check if we can import specific classes
    additional_checks = [
        ("langchain_aws", "BedrockEmbeddings", "Bedrock embeddings"),
        ("langchain_community.document_loaders", "PyPDFLoader", "PDF loader"),
        ("langchain_community.document_loaders", "Docx2txtLoader", "DOCX loader"),
        ("langchain_community.document_loaders", "TextLoader", "Text loader"),
        ("langchain_text_splitters", "RecursiveCharacterTextSplitter", "Text splitter"),
    ]
    
    for module_name, class_name, description in additional_checks:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name:30} - {description}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name:30} - {description}")
            print(f"   Error: {str(e)}")
    
    print()
    print("="*70)
    
    # Return results
    return {
        'statusCode': 200 if not results["missing_packages"] else 500,
        'body': json.dumps(results, indent=2)
    }


# For local testing
if __name__ == "__main__":
    result = lambda_handler({}, None)
    print("\nReturn value:")
    print(json.dumps(json.loads(result['body']), indent=2))
