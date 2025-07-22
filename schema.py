from pydantic import BaseModel

class TrainInput(BaseModel):
    speed: float
    signal_distance: float
    train_length: int
    train_speed_limit: int
    distance_to_next_train: float
    brake_applied: int
    time_to_next_signal: int
    signal_visible: int
    signal_status: str
    direction: str
    track_id: str
    weather_condition: str
