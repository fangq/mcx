# MCXLLM: natural-language prompt -> MCX input JSON (pydantic schema + langchain prompt,
# post-processed by LLMJson_post_gpt4o.py). Method: F.-Y. Yen, Y. Liu, and Q. Fang,
# J. Biomed. Opt. 31(11), 115003 (2026), doi:10.1117/1.JBO.31.11.115003. Called by
# mcxllm_bridge.py, which swaps in the model backend at run time; this file is otherwise
# the code evaluated in the paper.
import time
from pydantic import BaseModel, ValidationError
import json
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from openai import OpenAI

from LLMJson_post_gpt4o import parse_and_modify_json

from enum import Enum
from pydantic import BaseModel, conlist, field_validator, Field
from typing import List, Union, Optional, Tuple, Literal


class MediaItem(BaseModel):
    mua: float = 0
    mus: float = 0
    g: float = 1
    n: float = 1


class Sphere(BaseModel):
    O: Tuple[float, float, float]
    R: float
    Media: List[MediaItem]


class Box(BaseModel):
    O: Tuple[float, float, float]
    Size: Tuple[float, float, float]
    Media: List[MediaItem]


class Cylinder(BaseModel):
    C0: Tuple[float, float, float]
    C1: Tuple[float, float, float]
    R: float
    Media: List[MediaItem]


class Zlayers(BaseModel):
    Zlayers: Tuple[int, int] = Field(description="start and end")
    Media: List[MediaItem]


class Grid(BaseModel):
    Size: Tuple[int, int, int] = Field(description="background size")
    Media: List[MediaItem] = Field(description="background media properties")


# class Shapes(BaseModel):
#    shape: List[Union[Sphere, Box, Cylinder, Zlayers]] = None
class Exterior(BaseModel):
    Dim: Tuple[int, int, int] = [60, 60, 60]
    VolumeFile: str = ""
    LengthUnit: float = Field(default=1, description="Voxel Size (in mm)")
    Media: List[MediaItem] = Field(
        desicription="media outside the domain, use default value if not specified"
    )


class DetectorItem(BaseModel):
    Pos: Optional[Tuple[float, float, float]] = None
    R: Optional[float] = None


class Source(BaseModel):
    Type: Literal[
        "pencil", "isotropic", "cone", "gaussian", "hyperboloid", "planar", "disk"
    ] = Field(default="pencil")
    Pos: Tuple[float, float, float] = Field(default_factory=lambda: (30, 30, 30))
    Dir: Tuple[float, float, float] = Field(default_factory=lambda: (0, 0, 0))
    Param1: Tuple[float, float, float, float] = Field(
        default_factory=lambda: (0, 0, 0, 0)
    )
    Param2: Tuple[float, float, float, float] = Field(
        default_factory=lambda: (0, 0, 0, 0)
    )


class Optode(BaseModel):
    Source: Source
    Detector: List[DetectorItem]


class Forward(BaseModel):
    T0: float = 0
    T1: float = 5e-09
    Dt: float = 5e-09


class Session(BaseModel):
    ID: str
    Photons: float = 100000
    DoMismatch: bool = Field(
        default=True,
        description="set to False if user asked not to do reflection/transmission",
    )
    DoSaveRef: bool = Field(
        defaut=False, description="set to True if output type is diffuse reflectance"
    )
    # OutputType: OutputTypeEnum = OutputTypeEnum.output_flux
    OutputType: Literal[
        "output_flux",
        "fluence_rate",
        "fluence",
        "energy_deposit",
        "jacobian",
        "scattering",
        "momentum_transfer",
        "diffuse_reflectance",
    ] = Field(default="output_flux", description="Output type")


class MCXInput(BaseModel):
    Session: Session
    Forward: Forward
    Optode: Optode
    Shapes: List[Union[Sphere, Box, Cylinder, Zlayers]] = None
    # Shapes: Shapes
    Exterior: Exterior


param_instruct = """
'pencil' - default, pencil beam, no param needed
'isotropic' - isotropic source, no param needed
'cone' - uniform cone beam, Param1(1) is the half-angle in radian
'gaussian' [*] - a collimated gaussian beam, srcparam1(1) specifies the waist radius (in voxels)
'hyperboloid' [*] - a one-sheeted hyperboloid gaussian beam, Param1(1) specifies the waist
         radius (in voxels), Param1(2) specifies distance between launch plane and focus,
         Param1(3) specifies rayleigh range
'planar' [*] - a 3D quadrilateral uniform planar source. 
        Param1(1:3): Specify the vector Δ𝑥, which defines the direction and length of one edge of the quadrilateral starting from the origin (srcpos).
        Param2(1:3): Specify the vector Δy,which defines the direction and length of the other edge of the quadrilateral starting from the origin (srcpos).
'disk' [*] - a uniform disk source pointing along srcdir; the radius is 
        set by Param1(1) (in grid unit); if Param1(2) is set to a non-zero
        value, this source defines a ring (annulus) shaped source, with
        Param1(2) denoting the inner circle's radius, here Param1(1)>=Param1(2)
"""
####### Pydantic END ########
client = OpenAI()


def get_completion(prompt: str, model: str = "gpt-4o") -> str:
    # GPT4o model setup and get response
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=0, top_p=0
    )
    return completion.choices[0].message.content


def LLMMCX(query: str, original_output=0):
    # Set up the Pydantic parser
    pydantic_parser = PydanticOutputParser(pydantic_object=MCXInput)
    format_instructions = pydantic_parser.get_format_instructions()

    # Create the prompt template
    prompt = PromptTemplate(
        template="""Answer the user query. 
            Please make sure output is valid json and don't include shapes that is not mentioned. 
            Please do not create another box shape to describe the domain. 
            If Media parameters are not specified for the Domain, use the default values (mua=0, mus=0, g=1, n=1). 
            Under Shapes, each shape type (e.g., Sphere, Box, Cylinder) should appear as a distinct key containing a list of items. For example: "Shapes": [{{"Sphere": [...]}}, {{"Box": [...]}}]
            In the Shapes array, each shape type (e.g., Sphere, Box) can appear multiple times, each as a distinct key. For example:
            "Shapes": [
              {{ "Grid": {{Size: [60, 60, 60], "Media": [ {{...}} ] }} }},
              {{ "Sphere": {{ "O": [30, 30, 30], "R": 15, "Media": [ {{...}} ] }} }},
              {{ "Sphere": {{ "O": [40, 40, 40], "R": 20, "Media": [ {{...}} ] }} }},
              {{ "Box": {{ "O": [10, 10, 20], "Size": [10, 20, 30], "Media": [ {{...}} ] }} }}
            ]
            Please also note that "Grid" is always the first in the "Shapes" list, and add "Grid" with default values if user does not define the background properties.
            Do not group objects of the same type together in a list. Keep unrelated parts of the JSON unchanged.
            {format_instructions} 
            For the param parameters in source, here is a complete reference, for those who did not used, just keep the default values. If the query only provide a subset of required parameters, please use the default value.
            {param_instructions} {query}"
            """,
        input_variables=["query"],
        partial_variables={
            "format_instructions": pydantic_parser.get_format_instructions(),
            "param_instructions": param_instruct,
        },
    )

    # create input for LLM model
    _input = prompt.format_prompt(query=query)

    # Run LLM
    start_time = time.time()
    answer = get_completion(_input.to_string())
    end_time = time.time()

    try:
        # Parse and modify the JSON
        parsed_output = parse_and_modify_json(answer)
        if original_output == 0:
            return json.dumps(parsed_output)
        if original_output == 1:
            return json.dumps(parsed_output), answer  # return LLM raw output

    except Exception as e:
        if original_output == 0:
            return f"Error: {e}"
        if original_output == 1:
            return (
                f"Error: {e}. Original ouptut: {answer}",
                answer,
            )  # return LLM raw output
