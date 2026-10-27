import { Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, computed, inject, signal } from '@angular/core';
import { formatParameters } from '../common/model-architecture';
import type { ArchitectureGraph, ArchitectureNode } from '../types/modelArchitectureApi';

interface PlacedNode { node: ArchitectureNode; x: number; y: number; height: number; chips: string[]; }
interface PlacedEdge { path: string; label: string; labelX: number; labelY: number; cross: boolean; }

const NODE_WIDTH = 220;
const NODE_HEIGHT = 66;
const CHIP_ROW = 26;
const ROW_GAP = 46;
const LANE_GAP = 64;
const MARGIN = 14;
const MAX_CHIPS = 6;
// Narrower than this, the two lanes are stacked in one column.
const STACKED_BREAKPOINT = MARGIN * 2 + NODE_WIDTH * 2 + LANE_GAP;

// Presentation order of the graph nodes; the nodes themselves come from the API.
const LANE_ROWS: Record<string, { lane: 0 | 1; row: number }> = {
  image_input: { lane: 0, row: 0 },
  image_encoder: { lane: 0, row: 1 },
  projection: { lane: 0, row: 2 },
  encoder_stack: { lane: 0, row: 3 },
  token_input: { lane: 1, row: 0 },
  embedding: { lane: 1, row: 1 },
  decoder_stack: { lane: 1, row: 3 },
  classifier: { lane: 1, row: 4 },
};
const STACKED_ORDER = ['image_input', 'image_encoder', 'projection', 'encoder_stack', 'token_input', 'embedding', 'decoder_stack', 'classifier'];

export function stackChips(repeat: number, prefix: string): string[] {
  if (repeat <= MAX_CHIPS) return Array.from({ length: repeat }, (_, index) => `${prefix}${index + 1}`);
  return [`${prefix}1`, `${prefix}2`, '…', `${prefix}${repeat - 1}`, `${prefix}${repeat}`];
}

function formatShape(shape: (number | null)[]): string {
  return `[${shape.map((dim) => (dim === null ? 'B' : String(dim))).join(', ')}]`;
}

@Component({
  standalone: true,
  selector: 'app-architecture-diagram',
  styleUrl: '../styles/ArchitectureDiagram.css',
  template: `
    <svg class="arch-svg" [attr.viewBox]="'0 0 ' + layout().width + ' ' + layout().height" [attr.width]="layout().width" role="group" aria-label="Model architecture diagram">
      <defs>
        <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" class="arch-arrow-head" />
        </marker>
      </defs>
      @if (!stacked()) {
        <text class="arch-lane-title" [attr.x]="laneX(0)" y="14">Image path</text>
        <text class="arch-lane-title" [attr.x]="laneX(1)" y="14">Report path</text>
      }
      @for (edge of layout().edges; track edge.path) {
        <path class="arch-edge" [class.arch-edge-cross]="edge.cross" [attr.d]="edge.path" marker-end="url(#arch-arrow)" />
        @if (edge.cross && !stacked()) {
          <text class="arch-edge-label" [attr.x]="edge.labelX" [attr.y]="edge.labelY - 4" text-anchor="middle">{{ crossLabel(edge.label)[0] }}</text>
          <text class="arch-edge-label" [attr.x]="edge.labelX" [attr.y]="edge.labelY + 22" text-anchor="middle">{{ crossLabel(edge.label)[1] }}</text>
        } @else if (edge.label) { <text class="arch-edge-label" [attr.x]="edge.labelX" [attr.y]="edge.labelY">{{ edge.label }}</text> }
      }
      @for (placed of layout().nodes; track placed.node.id) {
        <g class="arch-node" [class.arch-node-input]="placed.node.kind === 'input'" [class.arch-node-frozen]="!placed.node.trainable && placed.node.parameters > 0" [class.selected]="placed.node.id === selected"
           [attr.transform]="'translate(' + placed.x + ',' + placed.y + ')'" tabindex="0" role="button"
           [attr.aria-pressed]="placed.node.id === selected" [attr.aria-label]="placed.node.label + ', ' + formatParameters(placed.node.parameters) + ' parameters'"
           (click)="nodeSelected.emit(placed.node.id)" (keydown.enter)="nodeSelected.emit(placed.node.id)" (keydown.space)="$event.preventDefault(); nodeSelected.emit(placed.node.id)">
          <rect class="arch-node-box" [attr.width]="nodeWidth" [attr.height]="placed.height" rx="10" />
          <text class="arch-node-label" x="14" y="24">{{ placed.node.label }}@if (placed.chips.length) { <tspan class="arch-node-repeat"> × {{ placed.node.repeat }}</tspan> }</text>
          @if (placed.node.parameters > 0) {
            <text class="arch-node-params" [attr.x]="nodeWidth - 14" y="24" text-anchor="end">{{ formatParameters(placed.node.parameters) }}</text>
          }
          <text class="arch-node-detail" x="14" y="44">{{ placed.node.detail }}</text>
          @if (!placed.node.trainable && placed.node.parameters > 0) {
            <text class="arch-node-badge" [attr.x]="nodeWidth - 14" y="44" text-anchor="end">frozen</text>
          }
          @for (chip of placed.chips; track $index; let index = $index) {
            <g [attr.transform]="'translate(' + (14 + index * 32) + ',' + (NODE_HEIGHT - 8) + ')'">
              <rect class="arch-chip" [class.arch-chip-gap]="chip === '…'" width="28" height="20" rx="5" />
              <text class="arch-chip-label" x="14" y="14" text-anchor="middle">{{ chip }}</text>
            </g>
          }
        </g>
      }
    </svg>
  `,
})
export class ArchitectureDiagramComponent implements OnInit, OnDestroy {
  private readonly host = inject(ElementRef<HTMLElement>);
  private observer: ResizeObserver | null = null;
  private readonly graphState = signal<ArchitectureGraph | null>(null);
  readonly stacked = signal(false);
  readonly nodeWidth = NODE_WIDTH;
  readonly NODE_HEIGHT = NODE_HEIGHT;
  readonly formatParameters = formatParameters;
  @Input() set graph(value: ArchitectureGraph | null) { this.graphState.set(value); }
  @Input() selected = '';
  @Output() readonly nodeSelected = new EventEmitter<string>();

  readonly layout = computed(() => {
    const graph = this.graphState();
    const stacked = this.stacked();
    const nodes: PlacedNode[] = [];
    if (!graph) return { width: 0, height: 0, nodes, edges: [] as PlacedEdge[] };

    const byId = new Map(graph.nodes.map((node) => [node.id, node]));
    const heightOf = (node: ArchitectureNode) => NODE_HEIGHT + ((node.repeat ?? 1) > 1 ? CHIP_ROW : 0);
    const chipsOf = (node: ArchitectureNode) => ((node.repeat ?? 1) > 1 ? stackChips(node.repeat ?? 1, node.kind === 'encoder_stack' ? 'E' : 'D') : []);
    const top = stacked ? MARGIN : MARGIN + 18;

    if (stacked) {
      let y = top;
      for (const id of STACKED_ORDER) {
        const node = byId.get(id);
        if (!node) continue;
        nodes.push({ node, x: MARGIN, y, height: heightOf(node), chips: chipsOf(node) });
        y += heightOf(node) + ROW_GAP;
      }
    } else {
      // Rows share a height so the encoder and decoder stacks line up.
      const rowHeights: number[] = [];
      for (const node of graph.nodes) {
        const slot = LANE_ROWS[node.id];
        if (slot) rowHeights[slot.row] = Math.max(rowHeights[slot.row] ?? 0, heightOf(node));
      }
      const rowY: number[] = [];
      let y = top;
      for (let row = 0; row < rowHeights.length; row += 1) { rowY[row] = y; y += (rowHeights[row] ?? NODE_HEIGHT) + ROW_GAP; }
      for (const node of graph.nodes) {
        const slot = LANE_ROWS[node.id];
        if (slot) nodes.push({ node, x: this.laneX(slot.lane), y: rowY[slot.row], height: heightOf(node), chips: chipsOf(node) });
      }
    }

    const placed = new Map(nodes.map((item) => [item.node.id, item]));
    const edges: PlacedEdge[] = [];
    for (const edge of graph.edges) {
      const source = placed.get(edge.source);
      const target = placed.get(edge.target);
      if (!source || !target) continue;
      const shape = formatShape(source.node.output_shape);
      if (edge.label) {
        if (stacked) {
          // Route the cross-attention edge along the right side of the column.
          const x = MARGIN + NODE_WIDTH;
          const sy = source.y + source.height / 2;
          const ty = target.y + target.height / 2;
          edges.push({ path: `M ${x} ${sy} H ${x + 24} V ${ty} H ${x + 2}`, label: edge.label, labelX: x + 30, labelY: (sy + ty) / 2, cross: true });
        } else {
          const y = target.y + target.height / 2;
          edges.push({ path: `M ${source.x + NODE_WIDTH} ${y} H ${target.x - 2}`, label: edge.label, labelX: source.x + NODE_WIDTH + LANE_GAP / 2, labelY: y - 4, cross: true });
        }
        continue;
      }
      const x = source.x + NODE_WIDTH / 2;
      const y1 = source.y + source.height;
      const y2 = target.y - 2;
      edges.push({ path: `M ${x} ${y1} V ${y2}`, label: shape, labelX: x + 8, labelY: (y1 + y2) / 2 + 4, cross: false });
    }

    const width = stacked ? MARGIN * 2 + NODE_WIDTH + 110 : MARGIN * 2 + NODE_WIDTH * 2 + LANE_GAP;
    const height = Math.max(...nodes.map((item) => item.y + item.height)) + MARGIN;
    return { width, height, nodes, edges };
  });

  laneX(lane: 0 | 1): number { return MARGIN + lane * (NODE_WIDTH + LANE_GAP); }

  /** Two lines that fit the gap between the lanes, e.g. cross- / attention. */
  crossLabel(label: string): [string, string] { const [first, ...rest] = label.split('-'); return rest.length ? [`${first}-`, rest.join('-')] : [label, '']; }

  ngOnInit(): void {
    if (typeof ResizeObserver === 'undefined') return;
    this.observer = new ResizeObserver(([entry]) => this.stacked.set(entry.contentRect.width < STACKED_BREAKPOINT));
    this.observer.observe(this.host.nativeElement);
  }

  ngOnDestroy(): void { this.observer?.disconnect(); }
}
