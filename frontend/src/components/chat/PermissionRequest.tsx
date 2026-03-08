import { useState } from 'react';
import { ShieldAlert, Check, X } from 'lucide-react';
import { useSessionStore } from '../../stores/sessionStore';
import { useChatStore } from '../../stores/chatStore';
import * as api from '../../api/client';

interface PermissionRequestProps {
  id: string;
  requestId: string;
  action: string;
  description: string;
  command: string;
  responded: boolean;
}

export function PermissionRequest(props: PermissionRequestProps) {
  const sessionId = useSessionStore((s) => s.activeSessionId);
  const [responding, setResponding] = useState(false);
  const [decision, setDecision] = useState<'approved' | 'denied' | null>(null);

  const handleRespond = async (approved: boolean) => {
    if (!sessionId || props.responded || responding) return;
    setResponding(true);
    try {
      await api.respondToPermission(sessionId, props.requestId, approved);
      setDecision(approved ? 'approved' : 'denied');
      useChatStore.setState((s) => ({
        pendingPermissions: s.pendingPermissions.map((p) =>
          p.id === props.id ? { ...p, responded: true } : p,
        ),
      }));
    } catch (err) {
      console.error('Permission response failed:', err);
    }
    setResponding(false);
  };

  return (
    <div className="flex gap-3 px-4 py-3">
      <div className="shrink-0 mt-0.5">
        <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
          <ShieldAlert className="w-4 h-4 text-amber-500" />
        </div>
      </div>
      <div className="min-w-0 flex-1 space-y-2">
        <div className="text-xs font-medium text-amber-500 uppercase tracking-wider">
          Permission Required
        </div>
        <p className="text-sm text-text-primary">{props.description}</p>
        <pre className="text-xs bg-bg-tertiary rounded-md px-3 py-2 overflow-x-auto text-text-secondary font-mono">
          {props.command}
        </pre>
        {decision ? (
          <div className={`flex items-center gap-1.5 text-xs font-medium ${
            decision === 'approved' ? 'text-green-400' : 'text-red-400'
          }`}>
            {decision === 'approved' ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
            {decision === 'approved' ? 'Approved' : 'Denied'}
          </div>
        ) : (
          <div className="flex gap-2">
            <button
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-green-600 hover:bg-green-500 text-white transition-colors disabled:opacity-50"
              onClick={() => handleRespond(true)}
              disabled={responding}
            >
              Approve
            </button>
            <button
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-bg-tertiary hover:bg-bg-secondary text-text-secondary border border-border-primary transition-colors disabled:opacity-50"
              onClick={() => handleRespond(false)}
              disabled={responding}
            >
              Deny
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
