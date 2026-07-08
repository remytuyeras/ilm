import re

def increment_version(name: str, semver_increment: str) -> str:
    # Regular expression to match the version number at the end
    pattern = r"(.*?)(\.v(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.pth)$"
    
    # Check if the model filename matches the pattern
    match = re.search(pattern, name)
    
    if match:
        # Extract the part before the version and the version components (x, y, z)
        base_name = match.group(1)
        x = int(match.group(3))
        y = int(match.group(4))
        z = int(match.group(5))
        
        # Increment version based on the semver argument
        if semver_increment == "major":
            x += 1
            y = 0
            z = 0
        elif semver_increment == "minor":
            y += 1
            z = 0
        elif semver_increment == "patch":
            z += 1
        else:
            raise ValueError("Invalid semver increment. Use 'major', 'minor', or 'patch'.")
        
        # Construct the new name with the updated version
        new_name = f"{base_name}.v{x}.{y}.{z}.pth"
    
    else:
        x = 0
        y = 0
        z = 0
        # Increment version based on the semver argument
        if semver_increment == "major":
            x = 1
        elif semver_increment == "minor":
            y = 1
        elif semver_increment == "patch":
            z = 1
        else:
            raise ValueError("Invalid semver increment. Use 'major', 'minor', or 'patch'.")
        
        # No version number found, add v0.0.0.pth
        new_name = name.replace(".pth",".v0.0.0.pth")
    
    return new_name
