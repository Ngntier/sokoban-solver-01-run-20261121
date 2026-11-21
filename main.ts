// A push-optimal Sokoban solver: one program answers a set of 5 levels in one run.
// Strategy: the generator builds every level as a hallway with independent rooms
// (3-4 boxes each) off it, and the level's push-optimum is the sum of the rooms'
// optima. So each room is solved push-optimally on its own (A* over pushes with
// a minimum-matching heuristic and deadlock pruning), the rooms in turn, the
// player walking between them. Pushes, not moves, are what the score counts:
// a level solved in P* pushes earns 1,000 points, one push over earns ~half.
//
// Deterministic by construction: fixed expansion budgets, no clock, no random
// numbers, ties broken by insertion order. The two executions the grader runs
// therefore write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four steps: [dx, dy, walk letter, push letter].
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// The most states one room search expands: work, never the clock. The generator
// keeps a room only when its own push-optimal search solves it within 250,000
// states, and A* with an admissible heuristic expands no more than that search
// does, so the budget is never the binding limit on a generated room.
const MAX_EXPAND = 250_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For every goal, the fewest pushes that bring a box from each square onto it,
 * other boxes ignored: a breadth-first search over box squares, pulling back
 * from the goal (a box at y can be pushed onto x when the player can stand at
 * y - d, i.e. both y and y - d are open floor). A square with no finite
 * distance to any goal is dead: a box there can never reach a goal.
 */
function pushDistances(level: Level): { dist: Int32Array[]; live: Uint8Array } {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const dist: Int32Array[] = goals.map(() => new Int32Array(size).fill(-1));
  goals.forEach((g, gi) => {
    const d = dist[gi]!;
    d[g] = 0;
    live[g] = 1;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const dd = dy * w + dx;
        const y = x - dd;
        if (y < 0 || y >= size || wall[y] || wall[y - dd] || d[y] >= 0) continue;
        d[y] = d[x]! + 1;
        live[y] = 1;
        queue.push(y);
      }
    }
  });
  return { dist, live };
}

/**
 * The squares the player reaches from `from` with `occupied` boxes in the way,
 * and the lowest of them: the player's reachable region, named by its lowest
 * square. Two states with the same boxes and the same region are one state.
 */
function regionLow(level: Level, occupied: Uint8Array, from: number): number {
  const { w, wall } = level;
  const size = wall.length;
  const seen = new Uint8Array(size);
  const stack = [from];
  seen[from] = 1;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= size || wall[q] || occupied[q] || seen[q]) continue;
      seen[q] = 1;
      stack.push(q);
    }
  }
  return low;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower LURD. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  const size = wall.length;
  if (from === to) return '';
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= size || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block (walls and boxes, a box off its goal among them). */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * Whether any box is frozen: it cannot be pushed in any direction, even
 * assuming the player can reach every open square, and it is off its goal.
 * Frozenness spreads: a box blocked only by frozen boxes is frozen too.
 * A frozen box off its goal can never move again, so the state is dead.
 */
function hasFrozenBox(level: Level, boxes: number[], goal: Uint8Array): boolean {
  const { w, wall } = level;
  const size = wall.length;
  const boxSet = new Uint8Array(size);
  for (const b of boxes) boxSet[b] = 1;
  const frozen = new Uint8Array(size);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen[b] || goal[b]) continue;
      let stuck = true;
      for (const [dx, dy] of DIRS) {
        const dd = dy * w + dx;
        const t = b + dd;
        const behind = b - dd;
        if (t < 0 || t >= size || behind < 0 || behind >= size) continue;
        // A push needs the target square free and the square behind it open.
        if (!wall[t] && !boxSet[t] && !wall[behind] && !frozen[behind]) { stuck = false; break; }
      }
      if (stuck) { frozen[b] = 1; changed = true; }
    }
  }
  return boxes.some((b) => frozen[b] && !goal[b]);
}

/**
 * A* over pushes: a state is the boxes' squares and the player's region.
 * The estimate is the cheapest assignment of boxes to goals by push-distance
 * (a minimum matching over the per-goal distances): each push moves one box
 * one square, so no solution from here costs fewer pushes than that sum.
 * Dead states are pruned (a box off the live squares, a frozen 2x2 block, a
 * frozen box). Returns {pushes, moves} (upper letters mark pushes) or null.
 */
function solvePushOptimal(level: Level, maxExpand = MAX_EXPAND): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = goals.length;
  if (n === 0) return { pushes: 0, moves: '' };
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { dist, live } = pushDistances(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  // The cheapest assignment of boxes to goals, by push-distance: every
  // permutation of the goals (at most 4! = 24), the cheapest total wins.
  // Infinity when some box reaches no goal: the state is dead. With more
  // than 6 goals the permutations explode, so the sum of each box's nearest
  // goal distance is used instead: weaker, but still a lower bound on the
  // pushes left (one push moves one box one square).
  const heuristic = (boxes: number[]): number => {
    if (n > 6) {
      let sum = 0;
      for (const b of boxes) {
        let m = Infinity;
        for (let j = 0; j < n; j++) { const d = dist[j]![b]; if (d >= 0 && d < m) m = d; }
        if (m === Infinity) return Infinity;
        sum += m;
      }
      return sum;
    }
    let best = Infinity;
    const assign = (i: number, used: number, cost: number): void => {
      if (cost >= best) return;
      if (i === n) { best = cost; return; }
      for (let j = 0; j < n; j++) {
        if (used & (1 << j)) continue;
        const d = dist[j]![boxes[i]!];
        if (d < 0) continue;
        assign(i + 1, used | (1 << j), cost + d);
      }
    };
    assign(0, 0, 0);
    return best;
  };

  // State key: the region's lowest square, then the boxes, as digits in base
  // size. A numeric key is exact while size ** (n + 1) stays under 2 ** 53
  // (a room of at most five boxes on any board this generator draws); beyond
  // that the key is a string, so a whole-level search with many boxes stays
  // exact too.
  const numericKeys = (n + 1) * Math.log2(size) < 52;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKeys) return low + ':' + boxes.join(',');
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const occupied = new Uint8Array(size);
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = regionLow(level, occupied, level.player);
  for (const b of level.boxes) occupied[b] = 0;

  // The states: boxes, the player's square after the push that made them,
  // the push (box from -> to) and the parent. The heap orders by f, then by
  // deeper g, then by insertion order: every tie is broken the same way on
  // every run.
  const sBoxes: number[][] = [level.boxes];
  const sPlayer: number[] = [level.player];
  const sKey: (number | string)[] = [keyOf(level.boxes, startLow)];
  const sPush: [number, number][] = [[-1, -1]];
  const parent: number[] = [-1];
  const gScore: number[] = [0];
  const bestG = new Map<number | string, number>([[keyOf(level.boxes, startLow), 0]]);
  const heap: [number, number, number, number][] = []; // [f, -g, seq, state]
  let seq = 0;
  const h0 = heuristic(level.boxes);
  if (h0 === Infinity) return null;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  let expanded = 0;
  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, , , s] = heapPop();
    const boxes = sBoxes[s]!;
    const g = gScore[s]!;
    // A stale heap entry: a better route to this state was recorded since.
    if (bestG.get(sKey[s]!) !== g) continue;
    expanded++;
    if (expanded > maxExpand) return null;
    for (const b of boxes) occupied[b] = 1;
    // The player's reachable region, as a stamp set, for push reachability.
    const reach = new Uint8Array(size);
    {
      const stack = [sPlayer[s]!];
      reach[sPlayer[s]!] = 1;
      while (stack.length > 0) {
        const p = stack.pop()!;
        for (const [dx, dy] of DIRS) {
          const q = p + dy * w + dx;
          if (q < 0 || q >= size || wall[q] || occupied[q] || reach[q]) continue;
          reach[q] = 1;
          stack.push(q);
        }
      }
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const dd = dy * w + dx;
        const t = b + dd;
        const behind = b - dd;
        if (t < 0 || t >= size || behind < 0 || behind >= size) continue;
        if (!reach[behind] || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          if (!hasFrozenBox(level, next, goal)) {
            key = keyOf(next, regionLow(level, occupied, b));
            h = heuristic(next);
          } else {
            next = null;
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (next === null || h === Infinity) continue;
        const ng = g + 1;
        const prev = bestG.get(key);
        if (prev !== undefined && prev <= ng) continue;
        bestG.set(key, ng);
        const ns = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sKey.push(key);
        sPush.push([b, t]);
        parent.push(s);
        gScore.push(ng);
        if (solved(next)) { found = ns; break; }
        heapPush([ng + h, -ng, seq++, ns]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // The path of pushes, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const [from, to] = sPush[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = boxes.map((b) => (b === from ? to : b)).sort((a, b2) => a - b2);
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room, its squares, boxes and goals. Null when some part
 * holds more or fewer boxes than goals, or a box starts in the hallway:
 * then the level is no set of rooms.
 */
function roomsOf(level: Level): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const { live } = pushDistances(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after `moves` (LURD, case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's.
 * Falls back to a search over the whole level when it is no set of rooms.
 */
function solveByRooms(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushOptimal(level);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done = new Set<{ squares: number[]; boxes: number[]; goals: number[] }>();
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of done.has(other) ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done.add(room);
  }
  return { pushes, moves };
}

/** One level's moves: room by room, or an empty answer when nothing solves it. */
function solve(text: string): string {
  const level = parseLevel(text);
  const solved = solveByRooms(level);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
