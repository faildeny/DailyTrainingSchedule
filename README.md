# DailyTrainingSchedule
Callback to pause Pytorch training during certain hours or days of a week.

When in sleep, the GPU memory will continue to be allocated, but it will stay silent and will require small amount of power.


```python
class TimeScheduleSleep(Callback):
    """
    Sleeps the training process during specified hours of the day

    Define hourly sleep schedule e.g. 8:00 - 21:00
    start_sleep_hour = 8
    end_sleep_hour = 21

    Define the last day to follow the schedule e.g. Friday
    0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday

    last_working_day = 5

    After this day, the model will not sleep until the end of the week

    """

    def __init__(self, start_sleep_hour=8, end_sleep_hour=21, last_working_day=5):
        self.start_sleep_hour = start_sleep_hour
        self.end_sleep_hour = end_sleep_hour
        self.last_working_day = last_working_day

    def on_train_batch_start(self, *args, **kwargs):
        current_time = datetime.now().strftime("%H")
        # Get day of the week
        day = datetime.now().weekday()
        current_time = int(current_time)
        # Follow the hourly schedule only until the last working day
        if day < self.last_working_day:
            if (
                current_time >= self.start_sleep_hour
                and current_time < self.end_sleep_hour
            ):
                print("Day and hour is: " + str(day) + " " + str(current_time))
                # Sleep until end of office hours
                time_to_sleep = self.end_sleep_hour - current_time
                print("Going to sleep for " + str(time_to_sleep) + " hours")
                time.sleep(time_to_sleep * 60 * 60)


```
