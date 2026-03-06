import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandShortcut,
  CommandSeparator,
} from '@/components/ui/command';
import {
  MessageSquare,
  Database,
  GitBranch,
  Settings,
  Plus,
  Download,
  Sun,
  Moon,
  Rocket,
  Keyboard,
} from 'lucide-react';
import { useSessionStore } from '../../stores/sessionStore';
import { useChatStore } from '../../stores/chatStore';
import * as api from '../../api/client';
import { exportSessionAsMarkdown } from '../../utils/exportSession';

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onOpenShortcuts?: () => void;
}

const QUICK_PROMPTS = [
  "Show me Parker Solar Probe data from its latest perihelion",
  "Compare ACE and Wind magnetic field data",
  "What's the current position of Solar Orbiter?",
];

export function CommandPalette({ open, onOpenChange, theme, onToggleTheme, onOpenShortcuts }: Props) {
  const navigate = useNavigate();
  const { savedSessions, activeSessionId, tokenUsage } = useSessionStore();
  const { messages } = useChatStore();
  const [search, setSearch] = useState('');

  // Reset search on close
  useEffect(() => {
    if (!open) setSearch('');
  }, [open]);

  const runAction = (fn: () => void) => {
    fn();
    onOpenChange(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Type a command or search..."
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => runAction(() => navigate('/'))}>
            <MessageSquare size={16} />
            Chat
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate('/data'))}>
            <Database size={16} />
            Data Tools
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate('/pipeline'))}>
            <GitBranch size={16} />
            Pipeline
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate('/settings'))}>
            <Settings size={16} />
            Settings
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Actions">
          <CommandItem onSelect={() => runAction(async () => {
            const oldId = useSessionStore.getState().activeSessionId;
            useChatStore.getState().clearChat();
            await useSessionStore.getState().createSession();
            if (oldId) api.deleteSession(oldId).catch(() => {});
            navigate('/');
          })}>
            <Plus size={16} />
            New Chat
            <CommandShortcut>Cmd+N</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runAction(onToggleTheme)}>
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            Toggle {theme === 'dark' ? 'Light' : 'Dark'} Mode
          </CommandItem>
          {onOpenShortcuts && (
            <CommandItem onSelect={() => runAction(onOpenShortcuts)}>
              <Keyboard size={16} />
              Keyboard Shortcuts
              <CommandShortcut>⌘?</CommandShortcut>
            </CommandItem>
          )}
          {messages.length > 0 && activeSessionId && (
            <CommandItem
              onSelect={() => {
                const session = savedSessions.find((s) => s.id === activeSessionId);
                runAction(() => {
                  exportSessionAsMarkdown(messages, activeSessionId, {
                    sessionName: session?.name,
                    tokenUsage,
                  });
                });
              }}
            >
              <Download size={16} />
              Export Session
            </CommandItem>
          )}
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Quick Prompts">
          {QUICK_PROMPTS.map((prompt) => (
            <CommandItem
              key={prompt}
              onSelect={() => {
                onOpenChange(false);
                navigate('/');
                // Small delay to allow navigation
                setTimeout(() => {
                  const { sendMessage } = useChatStore.getState();
                  const { activeSessionId: sid } = useSessionStore.getState();
                  if (sid) sendMessage(sid, prompt);
                }, 100);
              }}
            >
              <Rocket size={16} />
              {prompt}
            </CommandItem>
          ))}
        </CommandGroup>

        {savedSessions.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Recent Sessions">
              {savedSessions.slice(0, 5).map((s) => (
                <CommandItem
                  key={s.id}
                  onSelect={() => runAction(() => {
                    navigate('/');
                  })}
                >
                  <MessageSquare size={16} />
                  {s.last_message_preview?.slice(0, 60) || s.id.slice(0, 16)}
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
