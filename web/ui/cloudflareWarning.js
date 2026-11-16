export function showCloudflareWarning(extension, masterHost) {
    const existingBanner = document.getElementById('cloudflare-warning-banner');
    if (existingBanner) {
        existingBanner.remove();
    }

    const banner = document.createElement('div');
    banner.id = 'cloudflare-warning-banner';
    banner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #ff9800;
        color: #333;
        padding: 8px 16px;
        text-align: center;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;

    const messageSpan = document.createElement('span');
    // masterHost comes from config; build the message from text nodes.
    const hostStrong = document.createElement('strong');
    hostStrong.textContent = String(masterHost ?? '');
    messageSpan.append(
        'Connection issue: Master address ',
        hostStrong,
        ' is not reachable. The cloudflare tunnel may be offline.',
    );
    messageSpan.style.fontSize = '13px';

    const resetButton = document.createElement('button');
    resetButton.textContent = 'Reset Master Address';
    resetButton.style.cssText = `
        background: #333;
        color: white;
        border: none;
        padding: 6px 14px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: 500;
        font-size: 13px;
        transition: background 0.2s;
    `;
    resetButton.onmouseover = () => {
        resetButton.style.background = '#555';
    };
    resetButton.onmouseout = () => {
        resetButton.style.background = '#333';
    };

    const dismissButton = document.createElement('button');
    dismissButton.textContent = 'Dismiss';
    dismissButton.style.cssText = `
        background: transparent;
        color: #333;
        border: 1px solid #333;
        padding: 6px 14px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 13px;
        transition: opacity 0.2s;
    `;
    dismissButton.onmouseover = () => {
        dismissButton.style.opacity = '0.7';
    };
    dismissButton.onmouseout = () => {
        dismissButton.style.opacity = '1';
    };

    resetButton.onclick = async () => {
        resetButton.disabled = true;
        resetButton.textContent = 'Resetting...';

        try {
            await extension.api.updateMaster({
                name: extension.config?.master?.name || "Master",
                host: "",
            });

            if (extension.config?.master) {
                extension.config.master.host = "";
            }

            await extension.detectMasterIP();
            await extension.loadConfig();

            const newMasterUrl = extension.getMasterUrl();
            extension.log(`Master host reset. New URL: ${newMasterUrl}`, "info");

            if (extension.panelElement) {
                const hostInput = document.getElementById('master-host');
                if (hostInput) {
                    hostInput.value = extension.config?.master?.host || "";
                }
            }

            extension.app.extensionManager.toast.add({
                severity: "success",
                summary: "Master Host Reset",
                detail: `New address: ${newMasterUrl}`,
                life: 4000,
            });

            banner.remove();
        } catch (error) {
            resetButton.disabled = false;
            resetButton.textContent = 'Reset Master Host';
            extension.log(`Failed to reset master host: ${error.message}`, "error");
        }
    };

    dismissButton.onclick = () => {
        banner.remove();
    };

    banner.appendChild(messageSpan);
    banner.appendChild(resetButton);
    banner.appendChild(dismissButton);

    document.body.prepend(banner);

    setTimeout(() => {
        if (document.getElementById('cloudflare-warning-banner')) {
            banner.style.transition = 'opacity 0.5s';
            banner.style.opacity = '0';
            setTimeout(() => {
                banner.remove();
            }, 500);
        }
    }, 30000);
}
