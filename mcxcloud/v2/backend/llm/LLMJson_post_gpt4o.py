# Modified to process gpt 4o outputs:
# 1. change the order
# 2. Shape output is a list
import json
import copy


def find_and_parse_json(text):
    openings = [i for i, char in enumerate(text) if char == "{"]
    closings = [i for i, char in enumerate(text) if char == "}"]
    for start in openings:
        for end in reversed(closings):
            if start < end:  # Ensure valid order
                try:
                    possible_json_str = text[start : end + 1]
                    data = json.loads(possible_json_str)
                    return data  # Return the data if parsing is successful
                except json.JSONDecodeError:
                    continue  # Try the next possible substring
    return None  # Return None if no valid JSON was found


def replace_refs(shape_item):
    if "$ref" in shape_item:
        # Extract the shape type dynamically from the $ref (last part of the string after '/')
        shape_type = shape_item["$ref"].split("/")[-1]
        # Create a new shape structure dynamically, copying all the keys from shape_item
        shape_structure = {shape_type: {}}
        # Populate the new structure with all key-value pairs from the original shape_item, except "$ref"
        for key, value in shape_item.items():
            if key != "$ref":
                shape_structure[shape_type][key] = value
        # Clear the original item and update it with the dynamic shape structure
        shape_item.clear()
        shape_item.update(shape_structure)


def move_media_to_domain(data):
    # Create a deep copy of the data to avoid mutating the original input
    data = copy.deepcopy(data)
    domain_media = data["Domain"]["Media"]

    # Process shapes and extract media
    new_shapes = []
    for shape in data["Shapes"]:
        new_shape = {}
        for shape_type, shape_details in shape.items():
            if isinstance(shape_details, list):
                for shape_item in shape_details:
                    # Extract and move the Media if it exists
                    media = shape_item.pop("Media", None)
                    if media:
                        domain_media.extend(media)
            elif isinstance(shape_details, dict):
                # Special case for Zlayers
                media = shape_details.pop("Media", None)
                if media:
                    domain_media.extend(media)
                # Add the Zlayers to the new shape if it still has meaningful data
                if shape_details:
                    new_shape[shape_type] = shape_details
            else:
                # Keep other shape details unchanged
                new_shape[shape_type] = shape_details
        # Add the updated shape if it contains any meaningful data
        if new_shape:
            new_shapes.append(new_shape)

    # Replace the Shapes list with the updated version
    data["Shapes"] = new_shapes
    return data


def update_zlayer_for_diffuse_reflectance(data):
    # Check if DoSaveRef is true for diffuse reflectance
    if data["Session"].get("DoSaveRef", False):
        # Find `Zlayers` in `Shapes`
        for shape in data["Shapes"]:
            if "Zlayers" in shape:
                zlayers_data = shape["Zlayers"]
                zlayers = zlayers_data.get("Zlayers", [])
                # Check if thickness is 0
                if zlayers[1] - zlayers[0] == 0:
                    # Update the `Media` under `Zlayers` to be the same as `Media` under `Domain`
                    zlayers_data["Media"] = data["Domain"].get("Media", [])
    return data


def normalize_shapes(data):
    # Removing additional list under shape.sphere
    data = copy.deepcopy(data)
    for item in data["Shapes"]:
        for shape_type, shape_details in item.items():
            if isinstance(shape_details, list) and len(shape_details) == 1:
                item[shape_type] = shape_details[0]
    return data


# def reorder_keys(data):
#    key_order = ["Session", "Forward", "Optode", "Shapes", "Domain"]
#    reordered_data = {key: data[key] for key in key_order if key in data}
#    return reordered_data


def reorder_keys(data):
    """
    Reorders the given dictionary to match the specified structure.
    """
    ordered_structure = {
        "Session": {
            "ID": data.get("Session", {}).get("ID", ""),
            "Photons": data.get("Session", {}).get("Photons", 100000),
            "DoMismatch": data.get("Session", {}).get("DoMismatch", True),
            "DoSaveRef": data.get("Session", {}).get("DoSaveRef", False),
            "OutputType": data.get("Session", {}).get("OutputType", ""),
        },
        "Forward": data.get("Forward", {}),
        "Optode": data.get("Optode", {}),
        "Shapes": data.get("Shapes", []),
        "Domain": {
            "Dim": data.get("Domain", {}).get("Dim", []),
            "VolumeFile": data.get("Domain", {}).get("VolumeFile", ""),
            "LengthUnit": data.get("Domain", {}).get("LengthUnit", 1),
            "Media": data.get("Domain", {}).get("Media", []),
        },
    }
    return ordered_structure


def ensure_zlayers_in_shapes(data):
    # Create a deep copy of the data to avoid mutating the original input
    data = copy.deepcopy(data)
    shapes_list = data.get("Shapes", [])
    zlayers_shapes = []  # To collect all ZLayers shapes
    # Iterate over the Shapes list and extract Zlayers
    new_shapes_list = []
    for shape in shapes_list:
        if "Zlayers" in shape:
            zlayers_values = shape.pop("Zlayers", {}).get("Zlayers", None)
            if zlayers_values:
                # Transform each Zlayers into the required format
                zlayers_transformed = {"ZLayers": [zlayers_values]}
                zlayers_shapes.append(zlayers_transformed)
        # Add the shape to the new list if it's not empty
        if shape:
            new_shapes_list.append(shape)
    # Combine Zlayers shapes and the remaining shapes
    data["Shapes"] = zlayers_shapes + new_shapes_list
    return data


def ensure_zlayers_in_shapes(data):
    data = copy.deepcopy(data)  # Avoid mutating the original input
    shapes_list = data.get("Shapes", [])
    for shape in shapes_list:
        if "Zlayers" in shape:
            zlayers_values = shape.pop("Zlayers", {}).get("Zlayers", None)
            if zlayers_values:
                # Transform Zlayers into the required format
                shape["ZLayers"] = [zlayers_values]
    return data


def add_tags_to_shapes(data):
    # Add a 'Tag' field to each shape, and for 'ZLayers', append the Tag to the ZLayers list instead of using a separate field.
    data = copy.deepcopy(data)  # Avoid mutating the original input
    tag_counter = 1  # Initialize tag counter

    for shape in data.get("Shapes", []):
        for shape_type, shape_details in shape.items():
            if shape_type == "ZLayers":
                shape_details[0].append(tag_counter)
            else:
                # Add the tag as a separate field for other shapes
                shape_details["Tag"] = tag_counter
            tag_counter += 1
    return data


def map_output_type(data):
    output_type_mapping = {
        "output_flux": "x",
        "fluence_rate": "x",
        "fluence": "f",
        "energy_deposit": "e",
        "jacobian": "j",
        "scattering": "p",
        "momentum_transfer": "m",
        "diffuse_reflectance": "x",
    }

    data = copy.deepcopy(data)
    if "OutputType" in data.get("Session", {}):
        original_output_type = data["Session"]["OutputType"]
        data["Session"]["OutputType"] = output_type_mapping.get(
            original_output_type, original_output_type
        )
    return data


def ensure_json_keys(data):
    default_values = {
        "Session": {"Photons": 100000, "OutputType": "x"},
        "Forward": {
            "T0": 0,
            "T1": 5e-9,
            "Dt": 5e-9,
        },
        "Domain": {
            "Dim": [60, 60, 60],
            "VolumneFile": "",
            "LengthUnit": 1,
        },
    }
    # Check and add missing keys
    for key, defaults in default_values.items():
        if key not in data:
            data[key] = defaults  # Add the whole section if it's missing
        else:
            for subkey, value in defaults.items():
                if subkey not in data[key]:
                    data[key][
                        subkey
                    ] = value  # Add missing default values in the section
    return data


def parse_and_modify_json(text):
    text_without_backslashes = text.replace("\\", "")
    text_without_single_quote = text_without_backslashes.replace("'", '"')
    original_data = find_and_parse_json(text_without_single_quote)
    data = copy.deepcopy(original_data)
    data["Domain"] = data.pop("Exterior")  # Rename keys back to domain
    if "Shapes" in data and "shape" in data["Shapes"]:
        data["Shapes"] = data["Shapes"]["shape"]
        for shape_item in data["Shapes"]:
            replace_refs(shape_item)
    updated_diffuse_ref_data = update_zlayer_for_diffuse_reflectance(data)
    media_to_domain_data = move_media_to_domain(updated_diffuse_ref_data)
    normalized_data = normalize_shapes(media_to_domain_data)
    shapes_with_zlayers = ensure_zlayers_in_shapes(normalized_data)
    shapes_with_tags = add_tags_to_shapes(shapes_with_zlayers)
    mapped_data = map_output_type(shapes_with_tags)
    ensured_keys_data = ensure_json_keys(mapped_data)
    reordered_data = reorder_keys(ensured_keys_data)
    # ensure_media_data = ensure_correct_media(ensured_keys_data)
    return reordered_data
