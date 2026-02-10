#!/usr/bin/env python3
"""
Automated deployment script for the retrieval fix.
Run this on your local machine to deploy to the server.
"""
import subprocess
import sys
import time

SERVER_IP = "ec2-13-233-113-155.ap-south-1.compute.amazonaws.com"
SERVER_USER = "ubuntu"
SERVER_PATH = "~/ChatBotBE"
CONTAINER_NAME = "fastapi-backend"

def run_command(cmd, description, check=True):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if check and result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Success: {description}")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  RETRIEVAL FIX DEPLOYMENT SCRIPT                     ║
║                                                                      ║
║  This script will:                                                   ║
║  1. Commit and push changes to git                                   ║
║  2. SSH to server and pull latest code                               ║
║  3. Copy updated file to Docker container                            ║
║  4. Restart container                                                ║
║  5. Verify deployment                                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Git commit and push
    if not run_command(
        "git add rag_model/rag_utils.py diagnose_retrieval_issue.py fix_server_retrieval.sh RETRIEVAL_FIX_GUIDE.md deploy_retrieval_fix.py",
        "Adding files to git",
        check=False
    ):
        print("⚠️ Git add failed, but continuing...")
    
    if not run_command(
        'git commit -m "Fix: Increase topK multiplier to 50x for multi-tenant retrieval"',
        "Committing changes",
        check=False
    ):
        print("⚠️ Git commit failed (maybe no changes?), but continuing...")
    
    if not run_command(
        "git push origin main",
        "Pushing to remote",
        check=False
    ):
        print("⚠️ Git push failed, but continuing with local files...")
    
    # Step 2: SSH and pull on server
    ssh_pull_cmd = f'ssh {SERVER_USER}@{SERVER_IP} "cd {SERVER_PATH} && git pull origin main"'
    if not run_command(ssh_pull_cmd, "Pulling latest code on server", check=False):
        print("⚠️ Git pull on server failed, will copy file directly...")
    
    # Step 3: Copy file to container via server
    print("\n" + "="*70)
    print("📦 Copying updated file to container...")
    print("="*70)
    
    # First, copy to server
    scp_cmd = f"scp rag_model/rag_utils.py {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/rag_model/"
    if not run_command(scp_cmd, "Copying file to server", check=False):
        print("❌ SCP failed. Please copy manually:")
        print(f"   scp rag_model/rag_utils.py {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/rag_model/")
        sys.exit(1)
    
    # Then, copy from server to container
    docker_cp_cmd = f'ssh {SERVER_USER}@{SERVER_IP} "docker cp {SERVER_PATH}/rag_model/rag_utils.py {CONTAINER_NAME}:/app/rag_model/rag_utils.py"'
    if not run_command(docker_cp_cmd, "Copying file to container"):
        print("❌ Docker cp failed. Please run manually on server:")
        print(f"   docker cp {SERVER_PATH}/rag_model/rag_utils.py {CONTAINER_NAME}:/app/rag_model/rag_utils.py")
        sys.exit(1)
    
    # Step 4: Restart container
    restart_cmd = f'ssh {SERVER_USER}@{SERVER_IP} "docker restart {CONTAINER_NAME}"'
    if not run_command(restart_cmd, "Restarting container"):
        print("❌ Container restart failed")
        sys.exit(1)
    
    print("\n⏳ Waiting 10 seconds for container to restart...")
    time.sleep(10)
    
    # Step 5: Verify
    logs_cmd = f'ssh {SERVER_USER}@{SERVER_IP} "docker logs {CONTAINER_NAME} --tail 20"'
    run_command(logs_cmd, "Checking container logs", check=False)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     ✅ DEPLOYMENT COMPLETE                           ║
╚══════════════════════════════════════════════════════════════════════╝

Next Steps:
1. Test a query in your chatbot
2. Monitor logs: ssh {user}@{server} "docker logs -f {container}"
3. Look for: "Retrieved X documents for tenant 12 (checked Y vectors)"

Expected Output:
  ✅ Using S3 Vectors native filtering for tenant 12
  🔍 Retrieved 3-5 documents for tenant 12 (checked 3-5 vectors)

OR

  ⚠️ S3 Vectors filter failed (...), using manual filtering with larger topK
  🔍 Retrieved 3-5 documents for tenant 12 (checked 600 vectors)

If still getting 0 documents:
  - Run: python3 diagnose_retrieval_issue.py
  - Check: RETRIEVAL_FIX_GUIDE.md
    """.format(user=SERVER_USER, server=SERVER_IP, container=CONTAINER_NAME))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Deployment failed: {e}")
        sys.exit(1)
