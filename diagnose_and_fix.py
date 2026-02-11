#!/usr/bin/env python3
"""
Comprehensive diagnostic and fix script for fb_verify_token issue
This script will:
1. Diagnose the problem
2. Show you what's wrong
3. Offer to fix it automatically
"""

from database import engine, DATABASE_URL
from sqlalchemy import text, inspect
from models import Tenant
from database import SessionLocal
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def diagnose_and_fix():
    """Main diagnostic and fix function"""
    
    print_header("FB_VERIFY_TOKEN DIAGNOSTIC TOOL")
    print(f"\nDatabase: {DATABASE_URL}")
    
    db = SessionLocal()
    issues_found = []
    
    try:
        # Step 1: Check database type
        print_header("Step 1: Database Type")
        db_type = engine.dialect.name
        print(f"Database Type: {db_type}")
        
        if db_type not in ['postgresql', 'mysql', 'sqlite']:
            print(f"⚠️  Unsupported database type: {db_type}")
            return
        
        # Step 2: Check column definition
        print_header("Step 2: Column Definition Check")
        
        with engine.connect() as conn:
            if db_type == 'postgresql':
                result = conn.execute(text("""
                    SELECT 
                        column_name,
                        data_type,
                        character_maximum_length
                    FROM information_schema.columns 
                    WHERE table_name = 'tenants' 
                    AND column_name = 'fb_verify_token'
                """))
                
                row = result.fetchone()
                if row:
                    col_name, data_type, max_length = row
                    print(f"Column: {col_name}")
                    print(f"Type: {data_type}")
                    print(f"Max Length: {max_length if max_length else 'unlimited'}")
                    
                    if max_length and max_length < 50:
                        issues_found.append({
                            'type': 'column_too_short',
                            'current': max_length,
                            'required': 255
                        })
                        print(f"\n❌ ISSUE: Column is too short ({max_length} chars)")
                        print(f"   Required: At least 255 chars for tokens")
                    else:
                        print(f"\n✅ Column length is adequate")
                        
            elif db_type == 'sqlite':
                inspector = inspect(engine)
                columns = inspector.get_columns('tenants')
                for col in columns:
                    if col['name'] == 'fb_verify_token':
                        print(f"Column: {col['name']}")
                        print(f"Type: {col['type']}")
                        print(f"\n⚠️  SQLite doesn't enforce VARCHAR length")
                        print(f"   But the schema should still be correct")
        
        # Step 3: Check actual data
        print_header("Step 3: Data Check")
        
        tenants = db.query(Tenant).all()
        print(f"\nFound {len(tenants)} tenant(s)")
        print(f"\n{'ID':<5} {'Name':<25} {'Token':<30} {'Length':<8}")
        print("-" * 70)
        
        tenant_7 = None
        for tenant in tenants:
            token = tenant.fb_verify_token if tenant.fb_verify_token else "NULL"
            token_len = len(tenant.fb_verify_token) if tenant.fb_verify_token else 0
            token_display = (token[:27] + "...") if len(token) > 30 else token
            
            print(f"{tenant.id:<5} {tenant.name:<25} {token_display:<30} {token_len:<8}")
            
            if tenant.id == 7:
                tenant_7 = tenant
        
        # Step 4: Analyze tenant 7
        print_header("Step 4: Tenant 7 Analysis")
        
        if not tenant_7:
            print("❌ Tenant 7 not found in database")
            issues_found.append({'type': 'tenant_not_found'})
        else:
            print(f"Tenant Name: {tenant_7.name}")
            
            # Check verify token
            print(f"\nVerify Token:")
            print(f"  Current: '{tenant_7.fb_verify_token}'")
            print(f"  Length: {len(tenant_7.fb_verify_token) if tenant_7.fb_verify_token else 0}")
            
            expected_token = "chatbottoken@3420"
            if tenant_7.fb_verify_token == "d":
                print(f"  ❌ ISSUE: Token is truncated to 'd'")
                print(f"     Expected: '{expected_token}' (18 chars)")
                issues_found.append({
                    'type': 'verify_token_truncated',
                    'expected': expected_token,
                    'actual': tenant_7.fb_verify_token
                })
            elif not tenant_7.fb_verify_token:
                print(f"  ⚠️  Token is NULL/empty")
                issues_found.append({'type': 'verify_token_empty'})
            elif len(tenant_7.fb_verify_token) < 10:
                print(f"  ⚠️  Token seems too short")
                issues_found.append({
                    'type': 'verify_token_suspicious',
                    'actual': tenant_7.fb_verify_token
                })
            else:
                print(f"  ✅ Verify token looks OK")
            
            # Check access tokens
            print(f"\nFacebook Access Token:")
            fb_token_len = len(tenant_7.fb_access_token) if tenant_7.fb_access_token else 0
            print(f"  Length: {fb_token_len}")
            if tenant_7.fb_access_token:
                print(f"  Preview: {tenant_7.fb_access_token[:50]}...")
            
            if fb_token_len == 0:
                print(f"  ❌ ISSUE: Access token is NULL/empty")
                issues_found.append({'type': 'fb_access_token_empty'})
            elif fb_token_len < 100:
                print(f"  ❌ ISSUE: Access token is too short (truncated)")
                print(f"     Facebook tokens are typically 200-300 chars")
                issues_found.append({
                    'type': 'fb_access_token_truncated',
                    'length': fb_token_len
                })
            else:
                print(f"  ✅ Access token length looks OK")
            
            print(f"\nInstagram Access Token:")
            insta_token_len = len(tenant_7.insta_access_token) if tenant_7.insta_access_token else 0
            print(f"  Length: {insta_token_len}")
            if insta_token_len > 0 and insta_token_len < 100:
                print(f"  ⚠️  Token seems too short (may be truncated)")
                issues_found.append({
                    'type': 'insta_access_token_truncated',
                    'length': insta_token_len
                })
        
        # Step 5: Summary and fix options
        print_header("Step 5: Summary & Fix Options")
        
        if not issues_found:
            print("\n✅ No issues found!")
            print("\nIf you're still having webhook verification problems:")
            print("1. Check that the token in Facebook matches exactly")
            print("2. Check for URL encoding issues")
            print("3. Check nginx/proxy logs for request modification")
            return
        
        print(f"\nFound {len(issues_found)} issue(s):")
        for i, issue in enumerate(issues_found, 1):
            print(f"\n{i}. {issue['type']}")
            if issue['type'] == 'column_too_short':
                print(f"   Current length: {issue['current']}")
                print(f"   Required length: {issue['required']}")
            elif issue['type'] == 'verify_token_truncated':
                print(f"   Expected: {issue['expected']}")
                print(f"   Actual: {issue['actual']}")
            elif issue['type'] in ['fb_access_token_truncated', 'insta_access_token_truncated']:
                print(f"   Current length: {issue['length']}")
                print(f"   Required: 200+ characters for access tokens")
        
        # Offer fix
        if any(i['type'] == 'column_too_short' for i in issues_found):
            print("\n" + "=" * 70)
            print("FIX AVAILABLE")
            print("=" * 70)
            print("\nI can fix the column length issue automatically.")
            print("This will run:")
            if db_type == 'postgresql':
                print("  ALTER TABLE tenants ALTER COLUMN fb_verify_token TYPE VARCHAR(255);")
            elif db_type == 'mysql':
                print("  ALTER TABLE tenants MODIFY COLUMN fb_verify_token VARCHAR(255);")
            
            response = input("\nApply fix now? (yes/no): ")
            
            if response.lower() == 'yes':
                print("\nApplying fix...")
                with engine.connect() as conn:
                    if db_type == 'postgresql':
                        conn.execute(text("""
                            ALTER TABLE tenants ALTER COLUMN fb_verify_token TYPE VARCHAR(255)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants ALTER COLUMN fb_access_token TYPE VARCHAR(500)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants ALTER COLUMN insta_access_token TYPE VARCHAR(500)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants ALTER COLUMN telegram_bot_token TYPE VARCHAR(255)
                        """))
                    elif db_type == 'mysql':
                        conn.execute(text("""
                            ALTER TABLE tenants MODIFY COLUMN fb_verify_token VARCHAR(255)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants MODIFY COLUMN fb_access_token VARCHAR(500)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants MODIFY COLUMN insta_access_token VARCHAR(500)
                        """))
                        conn.execute(text("""
                            ALTER TABLE tenants MODIFY COLUMN telegram_bot_token VARCHAR(255)
                        """))
                    
                    conn.commit()
                
                print("\n✅ Fix applied successfully!")
                
                print("\n" + "=" * 70)
                print("NEXT STEPS")
                print("=" * 70)
                print("\n1. Restart your FastAPI application:")
                print("   cd /home/ubuntu/ChatBotBE")
                print("   docker-compose restart backend")
                print("\n2. Go to your admin UI and update tenant 7:")
                print("   - Set fb_verify_token to: chatbottoken@3420")
                print("   - Set fb_access_token to your FULL Facebook token")
                print("   - Set insta_access_token to your FULL Instagram token")
                print("\n3. Save the changes")
                print("\n4. Run this script again to verify:")
                print("   python diagnose_and_fix.py")
                print("\n5. Test webhook verification from Facebook")
                print("\n6. Test sending a message to your Facebook page")
            else:
                print("\nFix cancelled.")
                print("\nYou can run this script again anytime to apply the fix.")
        
        elif any(i['type'] in ['verify_token_truncated', 'verify_token_empty', 'verify_token_suspicious', 
                               'fb_access_token_truncated', 'fb_access_token_empty',
                               'insta_access_token_truncated'] for i in issues_found):
            print("\n" + "=" * 70)
            print("MANUAL ACTION REQUIRED")
            print("=" * 70)
            print("\nThe column length is OK, but the data needs to be updated.")
            print("\nSteps to fix:")
            print("1. Go to your admin UI")
            print("2. Edit tenant 7 (Organizations page)")
            print("3. Update the following:")
            
            if any(i['type'] in ['verify_token_truncated', 'verify_token_empty', 'verify_token_suspicious'] for i in issues_found):
                print("   - fb_verify_token: chatbottoken@3420")
            
            if any(i['type'] in ['fb_access_token_truncated', 'fb_access_token_empty'] for i in issues_found):
                print("   - fb_access_token: [Your FULL Facebook Page Access Token]")
                print("     (Get from: https://developers.facebook.com/)")
            
            if any(i['type'] == 'insta_access_token_truncated' for i in issues_found):
                print("   - insta_access_token: [Your FULL Instagram Access Token]")
            
            print("\n4. Save the changes")
            print("5. Run this script again to verify:")
            print("   python diagnose_and_fix.py")
            print("\n6. Test by sending a message to your Facebook page")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    try:
        diagnose_and_fix()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
