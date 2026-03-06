interface Props {
  size?: number;
  className?: string;
}

export function HelionLogo({ size = 28, className = '' }: Props) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={size}
      height={size}
      fill="none"
      className={className}
    >
      <circle cx="16" cy="16" r="10" fill="currentColor" opacity="0.9" />
      <circle cx="16" cy="16" r="6" fill="#0b1120" />
      <circle cx="16" cy="16" r="3.5" fill="currentColor" />
      {/* Corona rays */}
      <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.6">
        <line x1="16" y1="2" x2="16" y2="5" />
        <line x1="16" y1="27" x2="16" y2="30" />
        <line x1="2" y1="16" x2="5" y2="16" />
        <line x1="27" y1="16" x2="30" y2="16" />
        <line x1="6.1" y1="6.1" x2="8.2" y2="8.2" />
        <line x1="23.8" y1="23.8" x2="25.9" y2="25.9" />
        <line x1="6.1" y1="25.9" x2="8.2" y2="23.8" />
        <line x1="23.8" y1="8.2" x2="25.9" y2="6.1" />
      </g>
    </svg>
  );
}
