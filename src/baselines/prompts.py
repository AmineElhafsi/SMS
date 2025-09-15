SYSTEM_MESSAGE = "You are an assistant to an autonomous robot. Your job is to observe the robot's visual observations and answer its questions. The robot's query and relevant context will be provided under \"Input.\" More specific requirements pertaining to the query will be provided under \"Task.\" Please provide your response in the specified format."
LANDING_POSITION_QUERY_TEMPLATE = """Input:

The robot is a quadrotor looking to land on the {landing_target}. It needs to land on a stable position that will neither cause it to fall nor topple the landing platform or nearby objects.

Task:

An image of the quadrotor's current observation is provided. Landing position candidates are provided and are annotated on the image as circles with IDs. Which of these landing sites should the quadrotor choose as its landing position?

Please provide the output in the following format:

Reasoning: <e.g., What should the quadrotor consider? What are the risks? What are the safe areas?>
Decision: <ID of landing location; Please only specify the ID number>
"""
APPROACH_PATH_QUERY_TEMPLATE = """Input:

The robot is a quadrotor looking to land on the {landing_target}. It needs to land on a stable position that will neither cause it to fall nor topple the landing platform or nearby objects. It has identified a landing position and is now attempting to determine the best approach path. The approach path must account for the quadrotor propeller wash, which can impart a force on the objects below it, including the landing platform (i.e., the {landing_target}). The quadrotor should seek a path that minimally disturbs objects that it will fly over, and should especially try to avoid toppling the landing area.

Task:

An image of the quadrotor's current observation is provided. Approach path candidates are annotated as curves of different colors, each with a corresponding numerical ID. Note that these paths are projected to the height of the landing platform and the quadrotor would be flying at some small distance above these paths until it arrives at the landing position where it will descend. Which of these approach paths should the quadrotor take?

Please provide the output in the following format:

Reasoning: <e.g., What should the quadrotor consider? What are the risks?>
Decision: <ID of approach path; Please only specify the ID number>
"""

def parse_landing_position_response(response_text):
    """
    Parses the landing position query to extract the landing target ID.
    """
    # extract ID after "Decision:"
    landing_id_str = response_text.split("Decision:")[-1].strip()
    landing_id = int(landing_id_str)
    return landing_id

def parse_approach_path_response(response_text):
    """
    Parses the landing position query to extract the landing target ID.
    """
    # extract ID after "Decision:"
    approach_id_str = response_text.split("Decision:")[-1].strip()
    approach_id = int(approach_id_str)
    return approach_id


