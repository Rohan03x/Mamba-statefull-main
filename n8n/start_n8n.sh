#!/bin/bash
# ============================================================================
# n8n Startup Script for Mamba Training Pipeline
# ============================================================================
# Usage:
#   ./start_n8n.sh          # Start n8n stack
#   ./start_n8n.sh stop     # Stop n8n stack
#   ./start_n8n.sh logs     # View logs
#   ./start_n8n.sh shell    # Shell into n8n container
#   ./start_n8n.sh import   # Import workflow templates
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default workspace path
export MAMBA_WORKSPACE_PATH="${MAMBA_WORKSPACE_PATH:-$(dirname "$SCRIPT_DIR")}"

case "${1:-start}" in
    start)
        echo "🚀 Starting n8n stack..."
        echo "   Mamba workspace: $MAMBA_WORKSPACE_PATH"
        docker-compose up -d
        echo ""
        echo "✅ n8n is starting..."
        echo "   Web UI: http://localhost:5678"
        echo "   User: ${N8N_USER:-admin}"
        echo "   Password: ${N8N_PASSWORD:-mamba_n8n_password}"
        echo ""
        echo "   Wait ~30 seconds for all services to be ready."
        ;;
    
    stop)
        echo "🛑 Stopping n8n stack..."
        docker-compose down
        echo "✅ n8n stopped."
        ;;
    
    restart)
        echo "🔄 Restarting n8n stack..."
        docker-compose restart
        echo "✅ n8n restarted."
        ;;
    
    logs)
        docker-compose logs -f "${2:-n8n}"
        ;;
    
    shell)
        echo "🐚 Opening shell in n8n container..."
        docker-compose exec n8n /bin/sh
        ;;
    
    import)
        echo "📥 Importing workflow templates..."
        # Wait for n8n to be ready
        echo "   Waiting for n8n to be ready..."
        sleep 10
        
        # Import workflows using n8n CLI
        for workflow in workflows/*.json; do
            if [ -f "$workflow" ]; then
                echo "   Importing: $workflow"
                docker-compose exec -T n8n n8n import:workflow --input=/mamba_workspace/n8n/$workflow 2>/dev/null || \
                    echo "   (Import via CLI failed, use Web UI instead)"
            fi
        done
        
        echo ""
        echo "✅ Import complete. Check the n8n Web UI for imported workflows."
        echo "   Web UI: http://localhost:5678"
        ;;
    
    status)
        echo "📊 n8n Stack Status:"
        docker-compose ps
        ;;
    
    clean)
        echo "🧹 Cleaning up n8n data (WARNING: This deletes all workflows!)..."
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            echo "✅ Cleaned up."
        else
            echo "Cancelled."
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|logs|shell|import|status|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start n8n stack"
        echo "  stop    - Stop n8n stack"
        echo "  restart - Restart n8n stack"
        echo "  logs    - View n8n logs (optionally: logs postgres, logs redis)"
        echo "  shell   - Open shell in n8n container"
        echo "  import  - Import workflow templates"
        echo "  status  - Show container status"
        echo "  clean   - Remove all data (WARNING: destructive)"
        exit 1
        ;;
esac
