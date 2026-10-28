export interface OriginalSegment {
  text: string;
  changed: boolean;
}

const MAX_TOKENS = 4000;

/** Word-level diff that returns the original text split into runs flagged when absent from the current text. */
export function diffOriginal(original: string, current: string): OriginalSegment[] {
  const tokenize = (text: string) => text.split(/(\s+|[.,;:!?()])/).filter(Boolean);
  const before = tokenize(original);
  const after = tokenize(current);
  if (!before.length) return [];
  if (before.length > MAX_TOKENS || after.length > MAX_TOKENS) return [{ text: original, changed: false }];

  // Longest common subsequence table over tokens, then mark the original tokens that are not part of it.
  const rows = before.length + 1;
  const cols = after.length + 1;
  const lcs = Array.from({ length: rows }, () => new Uint16Array(cols));
  for (let i = before.length - 1; i >= 0; i -= 1) {
    for (let j = after.length - 1; j >= 0; j -= 1) {
      lcs[i][j] = before[i] === after[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const changed = new Array<boolean>(before.length).fill(true);
  let i = 0;
  let j = 0;
  while (i < before.length && j < after.length) {
    if (before[i] === after[j]) {
      changed[i] = false;
      i += 1;
      j += 1;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) i += 1;
    else j += 1;
  }

  // Whitespace only counts as changed when it sits between two changed words, so highlights form continuous runs.
  const isSpace = (token: string) => /^\s+$/.test(token);
  for (let k = 0; k < before.length; k += 1) {
    if (isSpace(before[k])) changed[k] = k > 0 && k < before.length - 1 && changed[k - 1] && changed[k + 1] && !isSpace(before[k - 1]);
  }

  const segments: OriginalSegment[] = [];
  before.forEach((text, index) => {
    const last = segments[segments.length - 1];
    if (last && last.changed === changed[index]) last.text += text;
    else segments.push({ text, changed: changed[index] });
  });
  return segments;
}
