<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import type { ThreadConfigModel } from "../apis/threads";
import { getModelName } from "../util";
import { checkAuthStatus, loginWithOAuth, logout, getToken } from "../apis/auth";

const props = defineProps<{
  model: ThreadConfigModel | null;
}>();

const emit = defineEmits<{
  (e: "update:model", value: ThreadConfigModel | null): void;
}>();

const useDefaultModel = ref(true);
const model = ref("");
const apiKey = ref("");
const isAuthenticated = ref(false);
const authLoading = ref(false);
const authError = ref("");
const maskedToken = ref("");

const modelOptions = [
  "gpt-5",
  "gpt-5.2",
  "gpt-5.3",
  "gpt-5.4",
  "gpt-5-mini",
  "gpt-5-nano",
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
  "gpt-4o",
  "gpt-4o-mini",
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
].map((m) => ({ label: m, value: getModelName(m) }));

watch(
  props,
  (newProps) => {
    if (newProps.model) {
      model.value = newProps.model.name ? newProps.model.name : modelOptions[0].value;
      apiKey.value = newProps.model.api_key || "";
      useDefaultModel.value = false;
    } else {
      model.value = "";
      apiKey.value = "";
      useDefaultModel.value = true;
    }
  },
  { immediate: true },
);

// Check authentication status on mount
onMounted(async () => {
  await checkAuth();
});

async function checkAuth() {
  try {
    const status = await checkAuthStatus();
    isAuthenticated.value = status.authenticated;
    if (status.token_info) {
      maskedToken.value = status.token_info.api_key;

      // Auto-fill API key if authenticated and current key is empty
      if (status.authenticated && !apiKey.value) {
        const tokenData = await getToken(true);
        if (tokenData.token) {
          apiKey.value = tokenData.token;
          console.log("Auto-loaded saved API key");

          // If using default model, auto-submit to save the token
          if (useDefaultModel.value && !props.model?.api_key) {
            console.log("Auto-submitting OAuth token to config");
            useDefaultModel.value = false;
            // Give UI a moment to update
            setTimeout(() => {
              submit();
            }, 100);
          }
        }
      }
    }
  } catch (error) {
    console.error("Failed to check auth status:", error);
  }
}

async function handleOAuthLogin() {
  authLoading.value = true;
  authError.value = "";

  try {
    const result = await loginWithOAuth(false);

    if (result.success) {
      // Fetch the actual token
      const tokenData = await getToken(true);
      if (tokenData.token) {
        apiKey.value = tokenData.token;
        await checkAuth();
        authError.value = "";
      }
    } else {
      authError.value = result.message;
    }
  } catch (error: any) {
    authError.value = error.message || "OAuth login failed";
    console.error("OAuth login error:", error);
  } finally {
    authLoading.value = false;
  }
}

async function handleLogout() {
  authLoading.value = true;
  authError.value = "";

  try {
    await logout();
    isAuthenticated.value = false;
    maskedToken.value = "";
    apiKey.value = "";
  } catch (error: any) {
    authError.value = error.message || "Logout failed";
    console.error("Logout error:", error);
  } finally {
    authLoading.value = false;
  }
}

function submit() {
  if (useDefaultModel.value) {
    emit("update:model", null);
  } else {
    emit("update:model", {
      name: model.value,
      api_key: apiKey.value,
    });
  }
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <FormItem for="defaultModel" label="Use Default Model" layout="row">
      <ToggleSwitch v-model="useDefaultModel" inputId="defaultModel" />
    </FormItem>

    <FormItem for="model" label="Model Configuration">
      <Select
        v-model="model"
        labelId="model"
        :options="modelOptions"
        optionLabel="label"
        optionValue="value"
        :disabled="useDefaultModel"
      />
    </FormItem>

    <FormItem for="apiKey" label="API Key">
      <div class="flex flex-col gap-3">
        <!-- OAuth status display -->
        <div v-if="isAuthenticated && maskedToken" class="flex items-center gap-2 text-sm">
          <span class="text-green-600">✓ Authenticated</span>
          <span class="text-gray-500">{{ maskedToken }}</span>
          <Button
            label="Logout"
            size="small"
            severity="secondary"
            @click="handleLogout"
            :disabled="authLoading || useDefaultModel"
          />
        </div>

        <!-- OAuth login button -->
        <Button
          v-if="!isAuthenticated"
          label="Login with OpenAI OAuth"
          icon="pi pi-sign-in"
          @click="handleOAuthLogin"
          :loading="authLoading"
          :disabled="useDefaultModel"
          severity="info"
        />

        <!-- Error message -->
        <div v-if="authError" class="text-red-600 text-sm">
          {{ authError }}
        </div>

        <!-- Manual API key input -->
        <div class="flex flex-col gap-1">
          <label class="text-sm text-gray-600">Or enter API key manually:</label>
          <InputText
            type="text"
            v-model="apiKey"
            inputId="apiKey"
            placeholder="Enter your API key"
            :disabled="useDefaultModel"
          />
        </div>
      </div>
    </FormItem>

    <Button label="Submit" @click="submit" />
  </div>
</template>
