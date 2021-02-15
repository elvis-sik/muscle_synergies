class Frequencies:
    _freq_forces_emg_section: int
    _freq_trajectories_section: int
    num_frames: int

    def __init__(self, frequency_forces_emg_section: int,
                 frequency_trajectories_sequence: int, num_frames: int):
        self._freq_forces_emg_section = frequency_forces_emg_section
        self._freq_trajectories_section = frequency_trajectories_sequence
        self.num_frames = num_frames

    @property
    def num_subframes(self) -> int:
        num = self._freq_forces_emg_section / self._freq_trajectories_section
        assert num == int(num)
        return int(num)

    def index(self, device_type: DeviceType, frame: int, subframe: int) -> int:
        self._validate_frame_arg(frame)
        self._validate_subframe_arg(subframe)

        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            return frame - 1
        return self._index_forces_emg(frame, subframe)

    def frame_subframe(self, device_type: DeviceType,
                       index: int) -> Tuple[int, int]:
        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            self._validate_traj_index_arg(index)
            return index + 1, 0

        self._validate_forces_emg_index_arg(index)
        return self._forces_emg_frame_subframe(index)

    def frame_range(self) -> range:
        return range(1, self.num_frames + 1)

    def subframe_range(self) -> range:
        return range(self.num_subframes)

    def frequency_of(self, device_type: DeviceType) -> int:
        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            return self._freq_trajectories_section
        return self._freq_forces_emg_section

    def _forces_emg_frame_subframe(self, index: int) -> Tuple[int, int]:
        # + 1 at the end because Python is 0-indexed
        frame = (index // self.num_subframes) + 1
        subframe = index % self.num_subframes
        return frame, subframe

    def _index_forces_emg(self, frame: int, subframe: int) -> int:
        return (frame - 1) * self.num_subframes + subframe

    def _validate_frame_arg(self, frame: int):
        if frame not in self.frame_range():
            raise ValueError(
                f'last frame is {self.num_frames}, frame {frame} is out of bounds'
            )

    def _validate_subframe_arg(self, subframe: int):
        if subframe not in self.subframe_range():
            raise ValueError(
                f'subframe {subframe} out of range {self.subframe_range()}')

    def _validate_traj_index_arg(self, index: int):
        final_index = self.num_frames - 1

        if index not in range(final_index + 1):
            raise ValueError(
                f'final index for trajectory marker is {final_index}, '
                'index {index} is out of bounds')

    def _validate_forces_emg_index_arg(self, index: int):
        final_index = self.num_frames * self.num_subframes - 1
        if index not in range(final_index + 1):
            raise ValueError(
                f'final index for force plates and EMG data is {final_index}, '
                f'index {index} out of bounds')


class DeviceHeaderData:
    device_name: str
    device_type: DeviceType
    _frequencies: Frequencies
    dataframe: pd.DataFrame

    def __init__(
            self,
            device_name: str,
            device_type: DeviceType,
            frequencies: Frequencies,
            dataframe: pd.DataFrame,
    ):
        self.device_name = device_name
        self.device_type = device_type
        self._frequencies = frequencies
        self.dataframe = dataframe

    @property
    def sampling_frequency(self) -> int:
        return self.frequencies.frequency_of(self.device_type)

    def slice_frame_subframe(self,
                             *,
                             stop_frame: int,
                             stop_subframe: int,
                             start_frame: Optional[int] = None,
                             start_subframe: Optional[int] = None,
                             step: Optional[int] = None) -> slice:
        stop_index = self._frequencies_index(stop_frame, stop_subframe)
        if start_frame is None:
            return slice(stop_index)

        start_index = self._frequencies_index(start_frame, start_subframe)
        if step is None:
            return slice(start_index, stop_index)
        return slice(start_index, stop_index, step)

    def _frequencies_index(self, frame: int, subframe: int) -> int:
        return self._frequencies.index(self.device_type, frame, subframe)

    @classmethod
    def from_device_header_pair(cls, device_header_pair: 'DeviceHeaderPair',
                                frequencies: Frequencies) -> 'DeviceHeader':
        device_name = device_header_pair.device_name
        device_type = device_header_pair.device_type
        dataframe = cls._device_header_pair_dataframe(device_header_pair)
        return cls(device_name=device_name,
                   device_type=device_type,
                   frequencies=frequencies,
                   dataframe=dataframe)

    @classmethod
    def _device_header_pair_dataframe(cls,
                                      device_header_pair: 'DeviceHeaderPair'
                                      ) -> pd.DataFrame:
        aggregator = device_header_pair.device_aggregator
        return cls._extract_dataframe(aggregator)

    @staticmethod
    def _extract_dataframe(device_header_aggregator: 'DeviceHeaderAggregator'
                           ) -> pd.DataFrame:
        def create_pint_array(data, physical_unit):
            PintArray(data, dtype=physical_unit)

        data_dict = {}
        for time_series_aggregator in device_header_aggregator:
            coord_name = time_series_aggregator.get_coordinate_name()
            physical_unit = time_series_aggregator.get_physical_unit()
            data = time_series_aggregator.get_data()
            data_dict[coord_name] = create_pint_parray(data, physical_unit)

        return pd.DataFrame(data_dict)


class ForcePlateData(DeviceHeaderData):
    def __init__(
            self,
            device_name: str,
            frequencies: Frequencies,
            dataframe: pd.DataFrame,
    ):
        super().__init__(device_name=device_name,
                         device_type=DeviceType.FORCE_PLATE,
                         frequencies=frequencies,
                         dataframe=dataframe)

    @classmethod
    def from_force_plate(cls, force_plate: 'ForcePlateDevices',
                         frequencies: Frequencies):
        device_name = force_plate.name

        force_device = force_plate.force
        moment_device = force_plate.moment
        cop_device = force_plate.cop

        force_dataframe = cls._device_header_pair_dataframe(force_device)
        moment_dataframe = cls._device_header_pair_dataframe(moment_device)
        cop_dataframe = cls._device_header_pair_dataframe(cop_device)

        dataframe = cls._join_dataframes(force_dataframe, moment_dataframe,
                                         cop_dataframe)

        cls(device_name=device_name,
            frequencies=frequencies,
            dataframe=dataframe)

    @staticmethod
    def _join_dataframes(*args: Tuple[pd.DataFrame]) -> pd.DataFrame:
        assert args

        if len(args) == 1:
            return args[0]
        return args[0].join(args[1:])


class DeviceMapping(collections.abc.Mapping, Generic[T]):
    device_list: List[Union[DeviceHeaderData, ForcePlateData]]
    devices_dict: Mapping[str, Union[DeviceHeaderData, ForcePlateData]]

    def __init__(
            self,
            device_list: List[T],
    ):
        self.device_list = list(device_list)
        self.devices_dict = self._build_devices_dict()

    def ith(self, i: int) -> T:
        return self.device_list[i]

    def _build_devices_dict(self):
        devices_dict = {}
        for device in device_list:
            device_name = device.device_name
            devices_dict[device_name] = device
        return devices_dict

    def __getitem__(self, device_name: str) -> pd.DataFrame:
        return self._devices_dict.__getitem__(ind)

    def __len__(self) -> int:
        return len(self._devices_dict)

    def __iter__(self) -> Iterable[str]:
        yield from iter(self._devices_dict)


@dataclass
class ViconNexusData:
    force_plates: Union[List[DeviceHeaderRepresentation],
                        ForcePlateRepresentation, 'DeviceMapping']
    emg: DeviceHeaderRepresentation
    trajectory_markers: List[
        Union[DeviceHeaderRepresentation, 'DeviceMapping']]

    def from_device_type(
            self, device_type: DeviceType
    ) -> Union[List[DeviceHeaderRepresentation], ForcePlateRepresentation,
               DeviceHeaderRepresentation, 'DeviceMapping']:
        if device_type is DeviceType.FORCE_PLATE:
            return self.force_plates
        if device_type is DeviceType.EMG:
            return self.emg
        if device_type is DeviceType.TRAJECTORY_MARKER:
            return self.trajectory_markers

        raise ValueError(f'device type {device_type} not understood')


@dataclass(frozen=True, eq=True)
class ForcePlateDevices:
    """The 3 device headers with the data of a single force plate.

    Since each force plate is represented in 3 different device headers, this
    class provides a standard way to represent those.

    Args:
        name: the name of a force plate, that is, everything happening in its
            column in the devices line before the dash (excluding the space).
            For example, the header
            'Imported AMTI OR6 Series Force Plate #1 - Force' should have name
            'Imported AMTI OR6 Series Force Plate #1'.

        force: the device referring to the force time series.

        moment: the device referring to the moment time series.

        cop: the device referring to the cop time series.
    """
    name: str
    force: DeviceHeaderRepresentation
    moment: DeviceHeaderRepresentation
    cop: DeviceHeaderRepresentation

    def list_devices(self) -> List[DeviceHeaderRepresentation]:
        return [self.force, self.moment, self.cop]
