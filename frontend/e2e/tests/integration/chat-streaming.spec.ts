import { test, expect } from '../../fixtures/base';
import { textReply, errorReply, toolCallReply, thinkingReply } from '../../mocks/sse-scenarios';

test.describe('@integration Chat streaming', () => {
  test.beforeEach(async ({ appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();
  });

  test('displays accumulated text from SSE text_delta events', async ({ chatPage }) => {
    // Send a message to trigger streaming state
    await chatPage.sendMessage('Hello');

    // Wait for user message to appear
    await expect(chatPage.getMessages('user').first()).toBeVisible();

    // Emit the full text reply scenario
    await chatPage.emitSSEScenario(textReply, 100);

    // Wait for agent message with the accumulated text
    await chatPage.waitForAgentMessage();
    const agentMessage = chatPage.getMessages('agent').first();
    await expect(agentMessage).toContainText('Hello!');
    await expect(agentMessage).toContainText('spacecraft data');
  });

  test('displays error message from SSE error event', async ({ chatPage }) => {
    await chatPage.sendMessage('Trigger error');
    await expect(chatPage.getMessages('user').first()).toBeVisible();

    // Emit error scenario
    await chatPage.emitSSEScenario(errorReply, 100);

    // Error should appear as an agent message with error text
    await chatPage.waitForAgentMessage();
    const agentMessage = chatPage.getMessages('agent').first();
    await expect(agentMessage).toContainText('Error');
  });

  test('shows tool call events in the activity area', async ({ page, chatPage }) => {
    await chatPage.sendMessage('Search for ACE data');
    await expect(chatPage.getMessages('user').first()).toBeVisible();

    // Emit tool call scenario
    await chatPage.emitSSEScenario(toolCallReply, 100);

    // Wait for agent response
    await chatPage.waitForAgentMessage();
    await expect(chatPage.getMessages('agent').first()).toContainText('ACE magnetic field');
  });

  test('shows thinking event as collapsible thinking message', async ({ page, chatPage }) => {
    await chatPage.sendMessage('Analyze this');
    await expect(chatPage.getMessages('user').first()).toBeVisible();

    // Emit thinking scenario
    await chatPage.emitSSEScenario(thinkingReply, 100);

    // Wait for agent response
    await chatPage.waitForAgentMessage();
    await expect(chatPage.getMessages('agent').first()).toContainText('Here is my analysis');
  });
});
