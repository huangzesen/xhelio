export function formatSessionDate(iso: string | null): string {
  if (!iso) return 'Unknown';
  const d = new Date(iso);
  const now = new Date();
  
  // Reset time for comparison
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  const dDate = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  
  if (dDate.getTime() === today.getTime()) return 'Today';
  if (dDate.getTime() === yesterday.getTime()) return 'Yesterday';
  
  const diffTime = today.getTime() - dDate.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays < 7) return 'This Week';
  if (diffDays < 30) return 'This Month';
  
  return 'Older';
}

export function getGroupOrder(group: string): number {
  switch (group) {
    case 'Pinned': return 0;
    case 'Today': return 1;
    case 'Yesterday': return 2;
    case 'This Week': return 3;
    case 'This Month': return 4;
    case 'Older': return 5;
    default: return 6;
  }
}

export function formatTokenCount(tokens: number): string {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
  return tokens.toString();
}

export function getTotalTokens(tokenUsage: Record<string, number>): number {
  if (tokenUsage.total_tokens != null) return tokenUsage.total_tokens;
  return (tokenUsage.input_tokens ?? 0) + (tokenUsage.output_tokens ?? 0) + (tokenUsage.thinking_tokens ?? 0);
}
