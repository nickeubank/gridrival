in the folder `~/github/gridrival` I'm working on creating a Python package for scenario planning for my fantasy formula 1 team on GridRival. In that folder in the `rules` folder you will find data on how scoring for the league works.

The first thing I want to do is write a script that takes a pandas data frame with the following columns as input and appends to that dataframe the resulting score for each driver.

For now, you can ignore constructors and sprint races — just think about drivers and grand prixs. Also don't worry about whether a driver finishes. You may assume all drivers finish.

Input dataframe:

- driver_name: unique identifier for driver
- driver_team: unique identifier for each team (each team has two drivers)
- eight_race_average: the driver's 8-race average finish position BEFORE the event being scored.
- starting_salary: the driver's salary before the event begins.
- qualifying_position: the hypothetical qualifying place in the current event.
- finishing_position: the hypothetical race finish position in the current event.

Output dataframe: input dataframe plus two columns:

- Points earned at event
- salary after event

claude --resume 0f589b2d-172c-402d-a6cc-7663ac5002e0
