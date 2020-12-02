from __future__ import annotations

import json
from typing import NamedTuple, List, Tuple, NewType, Optional, Any
import numpy as np
from enum import IntEnum


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


class PatternFingering:
    class FingerMapping(IntEnum):
        LEFT_PINKY = 1
        LEFT_RING = 2
        LEFT_MIDDLE = 3
        LEFT_INDEX = 4
        LEFT_THUMB = 5
        RIGHT_THUMB = 6
        RIGHT_INDEX = 7
        RIGHT_MIDDLE = 8
        RIGHT_RING = 9
        RIGHT_PINKY = 10
        UNKNOWN = 0

    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.fingering = np.zeros((4, 4), dtype=int)

    def set_finger_matrix(self, fingering: np.array):
        assert fingering.shape == (4, 4)
        for finger in np.nditer(fingering):
            _ = PatternFingering.FingerMapping(finger)
        self.fingering = np.copy(fingering)

    def get_finger_matrix(self) -> np.array:
        return np.copy(self.fingering)

    def to_numpy_arrays(
        self, *, pattern_kwargs: dict = None
    ) -> Tuple[np.array, np.array]:
        if pattern_kwargs is None:
            pattern_kwargs = dict()
        return self.pattern.to_numpy_array(**pattern_kwargs), self.get_finger_matrix()

    def to_json_dict(self) -> dict:
        """
        Serializes to a json-friendly dictionary that can be consumed by `json.dumps`.

        :return: json-friendly dictionary representation
        """
        obj = dict()
        obj["pattern"] = self.pattern.to_json_dict()
        obj["fingering"] = self.fingering.tolist()
        return obj

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> PatternFingering:
        """
        Deserializes from a json dict.

        :param json_dict: the json dictionary to read from
        :return: a new instance of Pattern
        """
        pattern = Pattern.from_json_dict(json_dict["pattern"])
        fingering = np.array(json_dict["fingering"], dtype=int)
        pf = PatternFingering(pattern)
        pf.set_finger_matrix(fingering)
        return pf

    def __repr__(self):
        return (self.pattern.to_numpy_array(), self.fingering).__repr__()


class TimedSequence:
    class TimeSelector:
        def get_start_index(self, ts: TimedSequence) -> Optional[int]:
            pass

        def get_end_index(self, ts: TimedSequence) -> Optional[int]:
            pass

    class Percent(TimeSelector):
        def __init__(self, percent: float):
            assert 0 <= percent <= 1
            self.percent = percent

        def get_start_index(self, ts: TimedSequence) -> Optional[int]:
            last_beat = ts.sequence[-1][0]
            min_beat = last_beat * self.percent
            for i, (beat_count, _) in enumerate(ts.sequence):
                if beat_count >= min_beat:
                    return i
            raise ValueError("This shouldn't happen, please file bug report")

        def get_end_index(self, ts: TimedSequence) -> Optional[int]:
            last_beat = ts.sequence[-1][0]
            max_beat = last_beat * self.percent
            for i, (beat_count, _) in enumerate(ts.sequence[::-1]):
                if beat_count <= max_beat:
                    return len(ts.sequence) - 1
            return len(ts.sequence) - 1

    class Beat(TimeSelector):
        def __init__(self, beat: float):
            assert beat >= 0
            self.beat = beat

        def get_start_index(self, ts: TimedSequence) -> Optional[int]:
            for i, (beat_count, _) in enumerate(ts.sequence):
                if beat_count >= self.beat:
                    return i
            raise ValueError("This shouldn't happen, please file bug report")

        def get_end_index(self, ts: TimedSequence) -> Optional[int]:
            for i, (beat_count, _) in enumerate(ts.sequence[::-1]):
                if beat_count <= self.beat:
                    return len(ts.sequence) - i
            return len(ts.sequence) - i

    class Second(TimeSelector):
        def __init__(self, seconds: float):
            assert seconds >= 0
            self.seconds = seconds

        def get_start_index(self, ts: TimedSequence) -> Optional[int]:
            last_beat = ts.sequence[-1][0]
            min_beat = ts.bpm / 60 * self.seconds
            if min_beat > last_beat:
                return None
            for i, (beat_count, _) in enumerate(ts.sequence):
                if beat_count >= min_beat:
                    return i
            raise ValueError("This shouldn't happen, please file bug report")

        def get_end_index(self, ts: TimedSequence) -> Optional[int]:
            last_beat = ts.sequence[-1][0]
            max_beat = ts.bpm / 60 * self.seconds
            if max_beat > last_beat:
                return None
            for i, (beat_count, _) in enumerate(ts.sequence[::-1]):
                if beat_count <= max_beat:
                    return len(ts.sequence) - i
            return len(ts.sequence) - i

    def __init__(self, sequence: List[Tuple[float, Any]], bpm: float):
        self.sequence = sequence
        self.bpm = bpm

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.sequence[key]
        if isinstance(key, slice):
            index_types = [type(index) for index in (key.start, key.stop)]
            # check the types are some sort of TimeSelector
            if not issubclass(
                index_types[0], (TimedSequence.TimeSelector, None.__class__)
            ) or not issubclass(
                index_types[1], (TimedSequence.TimeSelector, None.__class__)
            ):
                raise TypeError(
                    "Slice index types must be a subclass of TimedSequence.TimeSelector"
                )
            slice_start: TimedSequence.TimeSelector = key.start
            slice_end: TimedSequence.TimeSelector = key.stop
            index_start = None
            index_end = None
            if slice_start is not None:
                index_start = slice_start.get_start_index(self)
            if slice_end is not None:
                index_end = slice_end.get_end_index(self)
            return self.sequence[index_start:index_end]
        raise TypeError("Unsupported index type")

    def __len__(self):
        return self.sequence.__len__()

    def __delitem__(self, key):
        return self.sequence.__delitem__(key)

    def __setitem__(self, key, value):
        return self.sequence.__setitem__(key, value)

    def __iter__(self):
        return self.sequence.__iter__()

    def __reversed__(self):
        return self.sequence.__reversed__()

    def __contains__(self, item):
        return self.sequence.__contains__(item)


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


class JubeatChartFingering:
    def __init__(
        self,
        chart: JubeatChart,
        name: str,
        fingering_sequence: List[Tuple[float, PatternFingering]] = None,
    ):
        self.chart = chart
        self.name = name
        if fingering_sequence is None:
            self.fingering_sequence: List[Tuple[float, PatternFingering]] = [
                (beat_count, PatternFingering(pattern))
                for (beat_count, pattern) in self.chart.sequence
            ]
        else:
            self.fingering_sequence: List[
                Tuple[float, PatternFingering]
            ] = fingering_sequence

    def set_pattern_fingering(self, pattern_index: int, fingering: PatternFingering):
        self.fingering_sequence[pattern_index] = (
            self.fingering_sequence[pattern_index][0],
            fingering,
        )

    def to_json_dict(self) -> dict:
        """
        Serializes to a json-friendly dictionary that can be consumed by `json.dumps`.

        :return: json-friendly dictionary representation
        """
        obj = dict()
        obj["name"] = self.name
        obj["metadata"] = self.chart.metadata._asdict()
        obj["fingering_sequence"] = [
            {"beat_count": beat_count, **fingering.to_json_dict()}
            for (beat_count, fingering) in self.fingering_sequence
        ]
        return obj

    def to_json_string(self) -> str:
        """
        Serializes to a JSON string

        :return: JSON string representation
        """
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> JubeatChartFingering:
        """
        Deserializes from a json dict.

        :param json_dict: the json dictionary to read from
        :return: a new instance of Pattern
        """
        name = json_dict["name"]
        metadata = json_dict["metadata"]
        pattern_sequence = [
            (item["beat_count"], Pattern.from_json_dict(item["pattern"]))
            for item in json_dict["fingering"]
        ]
        chart = JubeatChart(metadata, pattern_sequence)
        fingering = [
            (
                item["beat_count"],
                PatternFingering.from_json_dict(
                    {"pattern": item["pattern"], "fingering": item["fingering"]}
                ),
            )
            for item in json_dict["fingering"]
        ]
        return JubeatChartFingering(chart, name, fingering)

    @classmethod
    def from_json_string(cls, json_str: str) -> JubeatChartFingering:
        """
        Deserializes from a JSON String

        :param json_str: JSON String representation
        :return: a new instance of JubeatChart
        """
        return cls.from_json_dict(json.loads(json_str))
