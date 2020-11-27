from __future__ import annotations

import json
from typing import NamedTuple, List, Tuple, NewType
import numpy as np


class Pattern:
    def __init__(self):
        self.data = set()
        self.hold_starts = set()
        self.hold_ticks = set()

    def add(self, position):
        """
        Adds a note to this position

        :param position: integer 0 ~ 15, starting from the top left hand corner
        """
        assert 0 <= position < 16
        assert position not in self.data
        assert position not in self.hold_starts
        assert position not in self.hold_ticks
        self.data.add(position)

    def add_hold(self, position):
        """
        Adds a hold note start to this position

        :param position: integer 0 ~ 15, starting from the top left hand corner
        """
        assert 0 <= position < 16
        assert position not in self.data
        assert position not in self.hold_starts
        assert position not in self.hold_ticks
        self.hold_starts.add(position)

    def add_hold_tick(self, position):
        """
        Adds a hold note tick to this position.
        A whole note tick means that a previous pattern in the sequence has a hold note in the same box and so this box is "unavailable".

        :param position: integer 0 ~ 15, starting from the top left hand corner
        """
        assert 0 <= position < 16
        assert position not in self.data
        assert position not in self.hold_starts
        assert position not in self.hold_ticks
        self.hold_ticks.add(position)

    def __repr__(self):
        pattern = []
        for r in range(0, 4):
            row = []
            for c in range(0, 4):
                position = (r * 4) + c
                if position in self.data:
                    row.append("1")
                elif position in self.hold_starts:
                    row.append("2")
                elif position in self.hold_ticks:
                    row.append("3")
                else:
                    row.append("0")
            pattern.append(row)
        return "\n" + "\n".join([" ".join(row) for row in pattern])

    def to_numpy_array(
        self, include_holds=False, differentiate_hold_start_and_tick: bool = True
    ) -> np.ndarray:
        """
        Returns a numpy array of the pattern as a 2D array

        :param include_holds: if false, all hold start and tick notes are 0
        :param differentiate_hold_start_and_tick: if `False`, both hold starts and hold ticks are '2', otherwise they are '2' and '3' respectively
        :return: 4x4 2D numpy array of ints
        """
        board_1d = []
        for i in range(16):
            if i in self.data:
                if i in self.hold_starts or i in self.hold_ticks:
                    if not include_holds:
                        board_1d.append(0)
                    else:
                        if i in self.hold_ticks and differentiate_hold_start_and_tick:
                            board_1d.append(3)
                        else:
                            board_1d.append(2)
                else:
                    board_1d.append(1)
            else:
                board_1d.append(0)
        return np.array(board_1d).reshape((4, 4))

    def to_json_dict(self) -> dict:
        """
        Serializes to a json-friendly dictionary that can be consumed by `json.dumps`.

        :return: json-friendly dictionary representation
        """
        return {
            "data": list(self.data),
            "hold_starts": list(self.hold_starts),
            "hold_ticks": list(self.hold_ticks),
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> Pattern:
        """
        Deserializes from a json dict.

        :param json_dict: the json dictionary to read from
        :return: a new instance of Pattern
        """
        p = Pattern()
        p.data = set(json_dict["data"])
        p.hold_starts = set(json_dict["hold_starts"])
        p.hold_ticks = set(json_dict["hold_ticks"])
        return p


class JubeatChart:
    # TODO: change these to literal types when keras supports 3.8
    TimeFormat = NewType("TimeFormat", str)
    TimeUnit = NewType("TimeUnit", str)

    class MetaData(NamedTuple):
        bpm: float
        time_signature: Tuple[int, int] = (4, 4)
        title: str = None
        artist: str = None
        chart: str = None
        difficulty: float = None

    def __init__(self, metadata: MetaData, sequence: List[Tuple[float, Pattern]]):
        self.metadata = metadata
        self.sequence = sequence

    def __repr__(self):
        repr = f"""
{self.metadata.title} / {self.metadata.artist}
{self.metadata.chart} - Lvl {self.metadata.difficulty}
{self.metadata.bpm}bpm ({self.metadata.time_signature[0]}/{self.metadata.time_signature[1]}) - {self.note_count} notes {"(contains holds)" if self.has_hold_notes else None}
        """
        return repr.strip()

    def to_numpy_array(
        self,
        include_holds: bool = True,
        differentiate_hold_start_and_tick: bool = True,
        time_format: TimeFormat = "delta",
        time_unit: TimeUnit = "seconds",
    ) -> np.ndarray:
        """
        Produces a numpy array that can be read by machine learning models.
        It produces an array of tuples, each representing a pattern occurring at a certain time. It is sorted chronologically.
        The first element of the tuple is the time code (float). It can either be in seconds or beats, and expressed as `absolute` or time to next pattern (`delta`). In the latter, the last pattern in the song will be coded as `np.inf`.
        The second element of the tuple a 2D array representing the pattern on the grid. Each cell is either 0 for empty space, 1 for normal note, and 2 for hold note (if not disabled).

        :param include_holds: if `False`, hold notes will not appear at all in the pattern data
        :param differentiate_hold_start_and_tick: if `False`, both hold starts and hold ticks are '2', otherwise they are '2' and '3' respectively
        :param time_format: `delta` or `absolute`
        :param time_unit: `seconds` or `beats`
        :return: numpy array
        """
        yeet = []
        for i, (beat_count, pattern) in enumerate(self.sequence):
            time_unit_value = (
                beat_count
                if time_unit == "beats"
                else beat_count / (self.metadata.bpm / 60)
            )
            if time_format == "absolute":
                yeet.append(
                    (
                        time_unit_value,
                        pattern.to_numpy_array(
                            include_holds=include_holds,
                            differentiate_hold_start_and_tick=differentiate_hold_start_and_tick,
                        ),
                    )
                )
            else:
                try:
                    next_entry = self.sequence[i + 1]
                    next_beat_count = next_entry[0]
                    next_time_unit_value = (
                        next_beat_count - beat_count
                        if time_unit == "beats"
                        else (next_beat_count - beat_count) / (self.metadata.bpm / 60)
                    )
                except IndexError:
                    next_time_unit_value = np.inf
                yeet.append(
                    (
                        next_time_unit_value,
                        pattern.to_numpy_array(include_holds=include_holds),
                    )
                )

        return np.array(yeet)

    @property
    def note_count(self) -> int:
        """
        Total number of notes for this song. DOES NOT INCLUDE HOLD TICKS.

        :return: note count
        """
        notes = 0
        for _, pattern in self.sequence:
            notes += len(pattern.data) + len(pattern.hold_starts)
        return notes

    @property
    def has_hold_notes(self) -> bool:
        """
        Checks whether this song has any hold notes present

        :return: whether hold notes are present
        """
        for _, pattern in self.sequence:
            if pattern.hold_starts:
                return True
        return False

    def to_json_dict(self) -> dict:
        """
        Serializes to a json-friendly dictionary that can be consumed by `json.dumps`.

        :return: json-friendly dictionary representation
        """
        obj = dict()
        obj["metadata"] = self.metadata._asdict()
        obj["sequence"] = [
            {"beat": beat, "pattern": pattern.to_json_dict()}
            for beat, pattern in self.sequence
        ]
        return obj

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> JubeatChart:
        """
        Deserializes from a json dict.

        :param json_dict: the json dictionary to read from
        :return: a new instance of JubeatChart
        """
        c = JubeatChart(
            metadata=JubeatChart.MetaData(**json_dict["metadata"]),
            sequence=[
                (obj["beat"], Pattern.from_json_dict(obj["pattern"]))
                for obj in json_dict["sequence"]
            ],
        )
        return c

    def to_json_string(self) -> str:
        """
        Serializes to a JSON string

        :return: JSON string representation
        """
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_json_string(cls, json_str: str) -> JubeatChart:
        """
        Deserializes from a JSON String

        :param json_str: JSON String representation
        :return: a new instance of JubeatChart
        """
        return cls.from_json_dict(json.loads(json_str))
