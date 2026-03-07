export const BUTTON_STYLES = {
    // Base styles with unified padding
    base: "width: 100%; padding: 4px 14px; color: white; border: none; border-radius: 4px; cursor: pointer; transition: all 0.2s; font-size: 12px; font-weight: 500;",
    
    // Context-specific combined styles
    workerControl: "flex: 1; font-size: 11px;",
    
    // Layout modifiers
    hidden: "display: none;",
    marginLeftAuto: "margin-left: auto;",
    
    // Color variants
    cancel: "background-color: #555;",
    info: "background-color: #333;",
    success: "background-color: #4a7c4a;",
    error: "background-color: #7c4a4a;",
    launch: "background-color: #4a7c4a;",
    stop: "background-color: #7c4a4a;",
    log: "background-color: #685434;",
    working: "background-color: #666;",
    clearMemory: "background-color: #555; padding: 6px 14px;",
    interrupt: "background-color: #555; padding: 6px 14px;",
};

export const STATUS_COLORS = {
    DISABLED_GRAY: "#666",
    OFFLINE_RED: "#c04c4c",
    ONLINE_GREEN: "#3ca03c",
    PROCESSING_YELLOW: "#f0ad4e"
};

export const UI_COLORS = {
    MUTED_TEXT: "#888",
    SECONDARY_TEXT: "#ccc",
    BORDER_LIGHT: "#555",
    BORDER_DARK: "#444",
    BORDER_DARKER: "#3a3a3a",
    BACKGROUND_DARK: "#2a2a2a",
    BACKGROUND_DARKER: "#1e1e1e",
    ICON_COLOR: "#666",
    ACCENT_COLOR: "#777"
};

export const PULSE_ANIMATION_CSS = `
    @keyframes pulse {
        0% {
            opacity: 1;
            transform: scale(0.8);
            box-shadow: 0 0 0 0 rgba(240, 173, 78, 0.7);
        }
        50% {
            opacity: 0.3;
            transform: scale(1.1);
            box-shadow: 0 0 0 6px rgba(240, 173, 78, 0);
        }
        100% {
            opacity: 1;
            transform: scale(0.8);
            box-shadow: 0 0 0 0 rgba(240, 173, 78, 0);
        }
    }
    .status-pulsing {
        animation: pulse 1.2s ease-in-out infinite;
        transform-origin: center;
    }

    .worker-status--online {
        background: var(--status-online, #3ca03c) !important;
    }

    .worker-status--offline {
        background: var(--status-offline, #c04c4c) !important;
    }

    .worker-status--unknown {
        background: var(--status-unknown, #888) !important;
    }

    .worker-status--processing {
        background: var(--status-processing, #f0ad4e) !important;
    }
    
    /* Button hover effects */
    .distributed-button:hover:not(:disabled) {
        filter: brightness(1.2);
        transition: filter 0.2s ease;
    }
    .distributed-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
    
    /* Settings button animation */
    .settings-btn {
        transition: transform 0.2s ease;
    }
    
    
    /* Expanded settings panel */
    .worker-settings {
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        transition: max-height 0.3s ease, opacity 0.3s ease, padding 0.3s ease, margin 0.3s ease;
    }
    .worker-settings.expanded {
        max-height: 500px;
        opacity: 1;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Cloudflare tunnel spinner */
    @keyframes tunnel-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .tunnel-spinner {
        width: 14px;
        height: 14px;
        border: 2px solid rgba(255, 255, 255, 0.35);
        border-top-color: #fff;
        border-radius: 50%;
        display: inline-block;
        animation: tunnel-spin 0.9s linear infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
`;

export const UI_STYLES = {
    statusDot: "display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px;",
    controlsDiv: "padding: 0 12px 12px 12px; display: flex; gap: 6px;",
    formGroup: "display: flex; flex-direction: column; gap: 5px;",
    formLabel: "font-size: 12px; color: var(--dist-label-text, #ccc); font-weight: 500;",
    formInput:
        "padding: 6px 10px; color: var(--dist-input-text, white); background: var(--dist-input-bg, transparent); font-size: 12px; transition: border-color 0.2s;",
    
    // Card styles
    cardBase: "margin-bottom: 12px; overflow: hidden; display: flex;",
    workerCard: "margin-bottom: 12px; overflow: hidden; display: flex;",
    cardBlueprint: "cursor: pointer; transition: all 0.2s ease;",
    cardAdd: "cursor: pointer; transition: all 0.2s ease;",

    // Column styles
    columnBase: "display: flex; align-items: center; justify-content: center;",
    checkboxColumn: "flex: 0 0 44px; display: flex; align-items: center; justify-content: center; cursor: default;",
    contentColumn: "flex: 1; display: flex; flex-direction: column; transition: background-color 0.2s ease;",
    iconColumn: "width: 44px; flex-shrink: 0; font-size: 20px; color: var(--dist-placeholder-add-color, #666);",
    
    // Row and content styles
    infoRow: "display: flex; align-items: center; padding: 12px; cursor: pointer; min-height: 64px;",
    workerContent: "display: flex; align-items: center; gap: 10px; flex: 1;",
    
    // Form and controls styles
    buttonGroup: "display: flex; gap: 4px; margin-top: 10px;",
    settingsForm: "display: flex; flex-direction: column; gap: 10px;",
    checkboxGroup: "display: flex; align-items: center; gap: 8px; margin: 5px 0;",
    formLabelClickable: "font-size: 12px; color: var(--dist-label-text, #ccc); cursor: pointer;",
    settingsToggle: "display: flex; align-items: center; gap: 6px; padding: 4px 0; cursor: pointer; user-select: none;",
    controlsWrapper: "display: flex; gap: 6px; align-items: stretch; width: 100%;",
    
    // Existing styles
    settingsArrow:
        "font-size: 12px; color: var(--dist-settings-arrow, #888); transition: all 0.2s ease; margin-left: auto; padding: 4px;",
    infoBox:
        "color: var(--dist-info-box-text, #999); padding: 5px 14px; font-size: 11px; text-align: center; flex: 1; font-weight: 500;",
    workerSettings: "margin: 0 12px; padding: 0 12px;"
};

export const TIMEOUTS = {
    DEFAULT_FETCH: 5000, // ms for general API calls
    STATUS_CHECK: 1200, // ms for status checks
    LAUNCH: 90000, // ms for worker launch (longer for model loading)
    RETRY_DELAY: 1000, // initial delay for exponential backoff
    MAX_RETRIES: 3, // max retry attempts
    REMOTE_LOG_AVAILABILITY_WINDOW: 30000, // show remote log button for this long after last online probe
    
    // UI feedback delays
    BUTTON_RESET: 3000, // button text/state reset after actions
    FLASH_SHORT: 1000, // brief success feedback
    FLASH_MEDIUM: 1500, // medium error feedback  
    FLASH_LONG: 2000, // longer error feedback
    
    // Operational delays
    POST_ACTION_DELAY: 500, // delay after operations before status checks
    STATUS_CHECK_DELAY: 100, // brief delay before status checks
    
    // Background tasks
    LOG_REFRESH: 2000, // log auto-refresh interval
    IMAGE_CACHE_CLEAR: 30000 // delay before clearing image cache
};

export const ENDPOINTS = {
    // ComfyUI core
    PROMPT: '/prompt',
    INTERRUPT: '/interrupt',
    UPLOAD_IMAGE: '/upload/image',
    SYSTEM_INFO: '/system_stats',

    // Distributed API
    CONFIG: '/distributed/config',
    UPDATE_WORKER: '/distributed/config/update_worker',
    DELETE_WORKER: '/distributed/config/delete_worker',
    UPDATE_SETTING: '/distributed/config/update_setting',
    UPDATE_MASTER: '/distributed/config/update_master',
    LAUNCH_WORKER: '/distributed/launch_worker',
    STOP_WORKER: '/distributed/stop_worker',
    MANAGED_WORKERS: '/distributed/managed_workers',
    WORKER_LOG: '/distributed/worker_log',
    REMOTE_WORKER_LOG: '/distributed/remote_worker_log',
    LOCAL_LOG: '/distributed/local_log',
    CLEAR_LAUNCHING: '/distributed/worker/clear_launching',
    PREPARE_JOB: '/distributed/prepare_job',
    LOAD_IMAGE: '/distributed/load_image',
    NETWORK_INFO: '/distributed/network_info',
    CHECK_FILE: '/distributed/check_file',
    CLEAR_MEMORY: '/distributed/clear_memory',
    SYSTEM_INFO_DIST: '/distributed/system_info',
    TUNNEL_START: '/distributed/tunnel/start',
    TUNNEL_STOP: '/distributed/tunnel/stop',
    TUNNEL_STATUS: '/distributed/tunnel/status',
};

export const NODE_CLASSES = {
    DISTRIBUTED_COLLECTOR: 'DistributedCollector',
    DISTRIBUTED_BRANCH: 'DistributedBranch',
    DISTRIBUTED_BRANCH_COLLECTOR: 'DistributedBranchCollector',
    DISTRIBUTED_SEED: 'DistributedSeed',
    DISTRIBUTED_EMPTY_IMAGE: 'DistributedEmptyImage',
    UPSCALE_DISTRIBUTED: 'UltimateSDUpscaleDistributed',
    PREVIEW_IMAGE: 'PreviewImage',
};

export function generateUUID() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}
