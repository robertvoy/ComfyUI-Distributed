import { describe, expect, it } from "vitest";
import { PULSE_ANIMATION_CSS } from "../constants.js";

describe("worker settings layout", () => {
    it("lets expanded forms grow instead of clipping the final controls", () => {
        const expanded = PULSE_ANIMATION_CSS.match(/\.worker-settings\.expanded\s*\{([^}]+)\}/)?.[1];
        expect(expanded).toBeDefined();
        expect(expanded).toMatch(/max-height:\s*none\s*;/);
    });

    it("still hides collapsed settings", () => {
        const collapsed = PULSE_ANIMATION_CSS.match(/\.worker-settings\s*\{([^}]+)\}/)?.[1];
        expect(collapsed).toMatch(/max-height:\s*0\s*;/);
        expect(collapsed).toMatch(/overflow:\s*hidden\s*;/);
    });
});
