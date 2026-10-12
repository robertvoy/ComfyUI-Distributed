import { afterEach, describe, expect, it, vi } from 'vitest';
vi.mock('../../scripts/api.js', () => ({ api: { clientId: 'test-client' } }), { virtual: true });
vi.mock('../workerLifecycle.js', () => ({
    checkAllWorkerStatuses: vi.fn(), getWorkerUrl: () => 'http://worker.invalid',
}));
vi.mock('../workerUtils.js', () => ({
    applyProbeResultToWorkerDot: vi.fn(),
    findNodesByClass: (prompt, cls) => Object.values(prompt).filter(node => node.class_type === cls),
}));
import { api } from '../../scripts/api.js';
import { setupInterceptor } from '../executionUtils.js';

const prompt = { output: { '1': { class_type: 'DistributedImageListStart', inputs: {} } } };
function extension(workers, online = true) {
    return {
        isEnabled: true, enabledWorkers: workers, log: vi.fn(), config: { settings: {} },
        originalQueuePrompt: vi.fn().mockResolvedValue({ prompt_id: 'local' }),
        api: { probeWorker: vi.fn().mockResolvedValue({ ok: online }),
            getServerlessStatus: vi.fn(),
            queueDistributed: vi.fn().mockResolvedValue({ prompt_id: 'distributed' }) },
        ui: { showToast: vi.fn() },
    };
}
afterEach(() => vi.useRealTimers());
describe('image-list queue interception', () => {
    it('routes image-list graphs through backend orchestration', async () => {
        vi.useFakeTimers();
        const ext = extension([{ id: 'one', type: 'local' }]);
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, prompt)).toEqual({ prompt_id: 'distributed' });
        expect(ext.api.queueDistributed).toHaveBeenCalledWith(expect.objectContaining({ enabled_worker_ids: ['one'] }));
        expect(ext.originalQueuePrompt).not.toHaveBeenCalled();
    });
    it('does not fall back when orchestrator-only workers are offline', async () => {
        vi.useFakeTimers();
        const ext = extension([{ id: 'one', type: 'local' }], false);
        ext.config.settings.master_delegate_only = true;
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, prompt)).toBeNull();
        expect(ext.originalQueuePrompt).not.toHaveBeenCalled();
        expect(ext.api.queueDistributed).not.toHaveBeenCalled();
    });
    it('does not probe serverless providers for an image-list job', async () => {
        vi.useFakeTimers();
        const ext = extension([{ id: 'pool', type: 'serverless' }]);
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, prompt)).toEqual({ prompt_id: 'distributed' });
        expect(ext.api.queueDistributed).toHaveBeenCalledWith(expect.objectContaining({ enabled_worker_ids: [], delegate_master: false }));
        expect(ext.api.getServerlessStatus).not.toHaveBeenCalled();
        expect(ext.originalQueuePrompt).not.toHaveBeenCalled();
    });
    it.each([{ workers: [] }, { workers: [{ id: 'one', type: 'local' }] }])('queues master-only through orchestration when no workers are online: $workers', async ({ workers }) => {
        vi.useFakeTimers();
        const ext = extension(workers, false);
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, prompt)).toEqual({ prompt_id: 'distributed' });
        expect(ext.api.queueDistributed).toHaveBeenCalledWith(expect.objectContaining({ enabled_worker_ids: [], delegate_master: false }));
        expect(ext.originalQueuePrompt).not.toHaveBeenCalled();
    });
    it('passes orchestrator-only mode to backend when workers are online', async () => {
        vi.useFakeTimers();
        const ext = extension([{ id: 'one', type: 'local' }]);
        ext.config.settings.master_delegate_only = true;
        setupInterceptor(ext);
        await api.queuePrompt(0, prompt);
        expect(ext.api.queueDistributed).toHaveBeenCalledWith(expect.objectContaining({ delegate_master: true }));
    });
    it('does not bypass backend errors with local execution', async () => {
        vi.useFakeTimers();
        const ext = extension([]);
        ext.api.queueDistributed.mockRejectedValue(new Error('invalid region'));
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, prompt)).toBeNull();
        expect(ext.api.queueDistributed).toHaveBeenCalled();
        expect(ext.originalQueuePrompt).not.toHaveBeenCalled();
    });
    it('preserves ordinary prompt handling', async () => {
        const ext = extension([]);
        setupInterceptor(ext);
        expect(await api.queuePrompt(0, { output: {} })).toEqual({ prompt_id: 'local' });
        expect(ext.api.queueDistributed).not.toHaveBeenCalled();
    });
});
