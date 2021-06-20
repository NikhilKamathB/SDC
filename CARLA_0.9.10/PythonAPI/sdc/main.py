import argparse
from config import Config as conf
from DATA_SCRIPTS.car_pose import car_pose

def main():
    pass

if __name__=="__main__":

    # Fetching arguments
    parser = argparse.ArgumentParser(description='SDC entry point.')
    # Level 1.
    parser.add_argument('-d', '--data', type=bool, default=False, metavar="\b", help='Is data generation/creation pipeline?')
    parser.add_argument('-s', '--simulation', type=bool, default=False, metavar="\b", help='Run the simulator?')
    # Global to level 2 and onwards.
    parser.add_argument('-c', '--configuration', type=str, default="./__DATA__", metavar="\b", help='Path to json file containing the Carla simulator configuration.')
    # Level 2 - data (-d).
    parser.add_argument('-cp', '--car_pose', type=bool, default=False, metavar="\b", help='Is the generation of data for car pose (--data must be True)?')
    parser.add_argument('-o', '--output_dir', type=str, default="./__DATA__", metavar="\b", help='Path to the output directory, the place where you want to store the outputs of data generators.')
    # Level 2 - simulation (-s).
    parser.add_argument('-v', '--verbose', type=bool, default=False, metavar="\b", help='Simulation verbose.')
    
    # Ramification of control.
    args = parser.parse_args()
    if args.data:
        if args.car_pose:
            # Get poses of a vehicle from various cameras spawned at different elivation and 
            car_pose(args.configuration, args.output_dir)
    elif args.simulation:
        pass
    else:
        print("Either --data (-d) or --simulation (-s) must be set. Both can't be unset.")