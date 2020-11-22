from io import TextIOBase
from pathlib import Path
from jubeatnet.core import JubeatChart, Pattern


class JubeatParser:
    def parse(self, filename: Path) -> JubeatChart:
        ...


class CosmosMemoParser(JubeatParser):
    class CharacterMap:
        NOTE_MAP = {
            "①": "1",
            "②": "2",
            "③": "3",
            "④": "4",
            "⑤": "5",
            "⑥": "6",
            "⑦": "7",
            "⑧": "8",
            "⑨": "9",
            "⑩": "10",
            "⑪": "11",
            "⑫": "12",
            "⑬": "13",
            "⑭": "14",
            "⑮": "15",
            "⑯": "16",
        }
        POSITION_MAP = {
            "口": "SQUARE",
            "＜": "HOLD_DIRECTION_LEFT",
            "＞": "HOLD_DIRECTION_RIGHT",
            "∨": "HOLD_DIRECTION_DOWN",
            "∧": "HOLD_DIRECTION_UP",
            "｜": "HOLD_STEM_VERTICAL",
            "―": "HOLD_STEM_HORIZONTAL",
        }
        MAP = {"|": "BAR", "ー": "EMPTY", **NOTE_MAP, **POSITION_MAP}
        NAMES = {v: k for k, v in MAP.items()}

    def __init__(self):
        self.buffer = None
        self.current_beat = 1

    def ingest(self, f: TextIOBase, *, unresolved_hold_notes=None):
        if unresolved_hold_notes is None:
            unresolved_hold_notes = set()
        # key is a tuple of note circle and board position
        # value is the following note circle it is resolved by
        hold_notes_resolved_by = {note: None for note in unresolved_hold_notes}
        note_positions_in_board = dict()
        note_positions_in_time = dict()
        # while set(note_positions_in_board.keys()) != set(note_positions_in_time.keys()) or not note_positions_in_board:
        lines = []
        board_lines_seen = 0
        rhythm_lines_seen = 0
        while board_lines_seen < 4 or rhythm_lines_seen < 4:
            l_raw = f.readline()
            if not l_raw:
                return None, unresolved_hold_notes
            l = l_raw.strip()
            if not l:
                continue
            if (
                l[0] in self.CharacterMap.POSITION_MAP
                or l[0] in self.CharacterMap.NOTE_MAP
            ):
                lines.append(l)
                board_lines_seen += 1
                if self.CharacterMap.NAMES["BAR"] in l:
                    rhythm_lines_seen += 1
        for l, line in enumerate(lines):
            pattern_section = line[:4]
            for i, note in enumerate(pattern_section):
                if note in self.CharacterMap.NOTE_MAP:
                    # dont append yet: we are not sure if this is actually end of hold note
                    unresolved_hold_notes = {
                        hold_note
                        for hold_note, v in hold_notes_resolved_by.items()
                        if v is None
                    }
                    position = ((l % 4) * 4) + i
                    is_ending_hold = False
                    for hold_note_start, hold_position in unresolved_hold_notes:
                        if position == hold_position:
                            # it is ending the previous hold note!
                            is_ending_hold = True
                            hold_notes_resolved_by[(hold_note_start, position)] = note
                            break
                    if not is_ending_hold:
                        positions = note_positions_in_board.get(note, list())
                        positions.append(position)
                        note_positions_in_board[note] = positions
                else:
                    continue
                # is it a hold note?
                # check left
                is_hold_note = False
                if i > 0:
                    valid_chars = {
                        self.CharacterMap.NAMES["HOLD_STEM_HORIZONTAL"],
                        self.CharacterMap.NAMES["HOLD_DIRECTION_RIGHT"],
                    }
                    if pattern_section[i - 1] in valid_chars:
                        is_hold_note = True
                # check right
                if i < 3:
                    valid_chars = {
                        self.CharacterMap.NAMES["HOLD_STEM_HORIZONTAL"],
                        self.CharacterMap.NAMES["HOLD_DIRECTION_LEFT"],
                    }
                    if pattern_section[i + 1] in valid_chars:
                        is_hold_note = True
                # check above
                if l % 4 > 0:
                    valid_chars = {
                        self.CharacterMap.NAMES["HOLD_STEM_VERTICAL"],
                        self.CharacterMap.NAMES["HOLD_DIRECTION_DOWN"],
                    }
                    above_pattern_section = lines[l - 1][:4]
                    if above_pattern_section[i] in valid_chars:
                        is_hold_note = True
                # check below
                if l % 4 < 3:
                    valid_chars = {
                        self.CharacterMap.NAMES["HOLD_STEM_VERTICAL"],
                        self.CharacterMap.NAMES["HOLD_DIRECTION_UP"],
                    }
                    below_pattern_section = lines[l - 1][:4]
                    if below_pattern_section[i] in valid_chars:
                        is_hold_note = True
                if is_hold_note:
                    position = ((l % 4) * 4) + i
                    # marked as unresolved
                    hold_notes_resolved_by[(note, position)] = None

            try:
                rhythm_section = line[4:].strip()
                increment_size = round(1 / (len(rhythm_section) - 2), 2)
                for t, note in enumerate(rhythm_section[1:-1]):
                    if note in self.CharacterMap.NOTE_MAP:
                        beat_number = self.current_beat + (l % 4) + (t * increment_size)
                        note_positions_in_time[note] = beat_number
            except IndexError:
                pass
        patterns = []
        for note, local_beat in note_positions_in_time.items():
            p = Pattern()
            for position in note_positions_in_board[note]:
                if (note, position) in hold_notes_resolved_by:
                    p.add_hold(position)
                    continue
                else:
                    p.add(position)
            for key, hold_end_note in hold_notes_resolved_by.items():
                hold_start_note, hold_position = key
                hold_start_beat = note_positions_in_time.get(hold_start_note, -1)
                hold_end_beat = note_positions_in_time.get(hold_end_note, local_beat)
                if hold_start_beat < local_beat <= hold_end_beat:
                    p.add_hold_tick(hold_position)
            patterns.append((local_beat, p))
        patterns.sort(key=lambda i: i[0])
        unresolved_hold_notes = {
            hold_note for hold_note, v in hold_notes_resolved_by.items() if v is None
        }
        return patterns, unresolved_hold_notes

    def parse(self, filename: Path) -> JubeatChart:
        f = filename.open()
        metadata = self.parse_header(f)
        unresolved_hold_notes = set()
        all_patterns = []
        while True:
            patterns, unresolved_hold_notes = self.ingest(
                f, unresolved_hold_notes=unresolved_hold_notes
            )
            if patterns is None:
                break
            all_patterns.extend(patterns)
            self.current_beat += 4
            unresolved_hold_notes = {
                ("-" + note[0], note[1]) for note in unresolved_hold_notes
            }
        f.close()
        return JubeatChart(sequence=all_patterns, metadata=metadata)

    def parse_header(self, f: TextIOBase) -> JubeatChart.MetaData:
        title = f.readline().strip()
        artist = f.readline().strip()
        f.readline()
        chart = f.readline().strip()
        f.readline()
        difficulty = float(f.readline().replace("Level:", "").strip())
        bpm = float(f.readline().replace("BPM:", "").strip())
        # note count
        f.readline()
        f.readline()
        return JubeatChart.MetaData(
            title=title, artist=artist, chart=chart, difficulty=difficulty, bpm=bpm
        )
