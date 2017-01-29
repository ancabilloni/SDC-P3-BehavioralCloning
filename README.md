# Behavioral Cloning for Self Driving Car
The scope of project is to teach car about human driving behavior so that the car can predict steering angle by itself. This is the 3rd project in Udacity Self Driving Car nanodegree. Data collection, driving and testing are performed on Udacity car simulation.
## Installation & Resources
1. Anaconda Python 3.5
2. Udacity [Carnd-term1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with miniconda installation 
3. Udacity Car Simulation on [Window x64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
4. Udacity [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

### Quickstart
**1. Control of the car is by using button on PC keyboard or joystick or game controller.**

:arrow_up: accelerate :arrow_down: brake :arrow_left: steer left :arrow_right: steer right

**2. Two driving modes:**
- Training: For user to take control over the car
- Autonomous: For car to drive by itself

**3. Collecting data:**
User drives on track 1 and collects data by recording the driving experience by toggle ON/OFF the recorder. Data is saved as frame images and a driving log which shows the location of the images, steering angle, throttle, speed, etc. 
Another option is trying on Udacity data sample.

## Test Drive
Drive around the tracks several time to feel familiar with the roads and observe the environment around the track.

Track 1: *flat road, mostly straight driving, occasionally sharp turns, bright day light.*
![track1](https://cloud.githubusercontent.com/assets/23693651/22400792/a8927a68-e58c-11e6-8a66-839869832cce.png)

Track 2: *hilly road, many light turns and sharp turns, dark environment*
![track2](https://cloud.githubusercontent.com/assets/23693651/22400796/be938938-e58c-11e6-9938-6ba32ef3d554.png)
### Understanding Data
There are 3 cameras on the car which shows left, center and right images for each steering angle. 
