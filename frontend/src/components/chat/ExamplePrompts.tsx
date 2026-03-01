import { motion } from 'framer-motion';
import { Orbit, Sun, Zap, Globe, Telescope, Satellite } from 'lucide-react';
import { HelionLogo } from '../common/HelionLogo';
import { fadeSlideIn, stagger } from '../common/MotionPresets';

const EXAMPLES = [
  {
    icon: Orbit,
    title: "Voyager 1's escape",
    prompt: "How did scientists prove Voyager 1 left the solar system? Show me the data.",
  },
  {
    icon: Sun,
    title: "PSP corona entry",
    prompt: "When did Parker Solar Probe first enter the solar corona? Show me what happened.",
  },
  {
    icon: Zap,
    title: "CME impact",
    prompt: "Show me a powerful coronal mass ejection hitting Earth. What did it look like in the data?",
  },
  {
    icon: Satellite,
    title: "PSP perihelion deep-dive",
    prompt: "Load recent Parker Solar Probe perihelion data ±3 days around the perihelion. Show a 7-panel plot: (1) Br, Bt, Bn, and |B|; (2) Vr and Br — convert Br to velocity units; (3) compare proton density between SPI and QTN; (4) cross helicity — use a 2-hour rolling average as background and calculate cross helicity with the background subtracted; (5) trace magnetic power spectrogram from wavelet, normalized by f (since P(f)·df = P(f)·f·d ln f), with a linear z-axis for geometric meaning; (6) compare proton, alpha-particle radial velocity, and Alfvén velocity; (7) high-cadence heliocentric distance. Use the 4-per-cycle magnetic field product. Keep colorbars from overlapping adjacent panels.",
  },
  {
    icon: Globe,
    title: "Solar storm comparison",
    prompt: "Compare ACE and Wind magnetic field and solar wind proton density during the September 2017 solar storm.",
  },
  {
    icon: Telescope,
    title: "Jupiter flyby",
    prompt: "Show me Juno's magnetic field as it dove through Jupiter's magnetosphere on its first perijove pass.",
  },
];

const SPACECRAFT = [
  'PSP', 'Solar Orbiter', 'ACE', 'Wind', 'DSCOVR', 'MMS',
  'STEREO-A', 'Juno', 'Voyager 1', 'Voyager 2', 'Cassini', 'MAVEN',
];

interface Props {
  onSelect: (prompt: string) => void;
}

export function ExamplePrompts({ onSelect }: Props) {
  return (
    <div data-testid="example-prompts" className="flex-1 flex flex-col items-center justify-center px-6 pb-4 overflow-y-auto">
      <motion.div
        className="flex flex-col items-center"
        variants={stagger}
        initial="hidden"
        animate="visible"
      >
        {/* Hero */}
        <motion.div variants={fadeSlideIn} className="flex flex-col items-center mb-8">
          <HelionLogo size={48} className="text-primary mb-4" />
          <h1 className="text-2xl font-bold text-text mb-2">
            Welcome to XHelio
          </h1>
          <p className="text-sm text-text-muted text-center max-w-md">
            Your AI-powered interface for spacecraft data exploration.
            Ask about solar events, planetary missions, or fetch and visualize data.
          </p>
        </motion.div>

        {/* Example cards */}
        <motion.div
          variants={stagger}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-w-3xl w-full mb-8"
        >
          {EXAMPLES.map(({ icon: Icon, title, prompt }) => (
            <motion.button
              data-testid="example-prompt"
              key={title}
              variants={fadeSlideIn}
              onClick={() => onSelect(prompt)}
              className="text-left p-4 rounded-xl border border-border
                hover:border-primary hover:bg-primary/5 transition-colors group"
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon size={16} className="text-primary" />
                <span className="text-sm font-medium text-text group-hover:text-primary transition-colors">
                  {title}
                </span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed line-clamp-2">
                {prompt}
              </p>
            </motion.button>
          ))}
        </motion.div>

        {/* Spacecraft showcase */}
        <motion.div variants={fadeSlideIn} className="text-center">
          <p className="text-xs text-text-muted mb-2">Supported spacecraft</p>
          <div className="flex flex-wrap justify-center gap-2 max-w-lg">
            {SPACECRAFT.map((name) => (
              <span
                key={name}
                className="px-2 py-0.5 rounded-full text-[11px] bg-surface-elevated text-text-muted border border-border"
              >
                {name}
              </span>
            ))}
          </div>
        </motion.div>

        {/* Keyboard hint */}
        <motion.div variants={fadeSlideIn} className="mt-6 text-center">
          <p className="text-[11px] text-text-muted">
            Press{' '}
            <kbd className="px-1.5 py-0.5 rounded bg-surface-elevated border border-border text-[10px] font-mono">
              Cmd+K
            </kbd>{' '}
            to open the command palette
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
