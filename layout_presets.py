"""
Graph Layout Presets Module

This module contains manual layout adjustments for graph visualization of module connections.
These presets improve node placement and label positioning for different configurations
(e.g., Europa, Mars, TRAPPIST-1e, etc.) to enhance readability.

Author: QHF Development Team
"""

# ======================================
# NODE POSITION OFFSETS
# ======================================

# Positional offsets to move nodes slightly for better readability
# Format: {habitat_name: {node_name: (x_offset, y_offset)}}
presets = {
    "europa": {
        "Equilibrium \n Temperature": (0.00, -0.15),
        "Semi major axis": (0.00, -0.25),
        "Leaky Greenhouse": (0.00, 0.20),
        "Planet\nPrimary\nProperties": (0.00, 0.15)
    },
    "mars": {
        "Equilibrium \n Temperature": (0.05, 0.05),
        "Semi major axis": (0.00, -0.15),
        "Leaky Greenhouse": (0.00, 0.10),
        "Planet\nPrimary\nProperties": (0.00, 0.12),
        "Methanogens AE v1.0": (0.00, 0.12),
        "Stellar \n Properties": (-0.12, -0.05),
        "Albedo Prior": (-0.10, -0.05),
        "Luminosity": (0.10, 0.10),
        "Bond Albedo": (0.15, -0.10),
        "Stellar Mass": (-0.15, 0.10)
    },
    
    "test_visexoplanet": {
        "Albedo Prior": (0.00, -0.20),
        "Equilibrium \n Temperature": (0.15, 0.10),
        "Greenhouse \n Effect": (0.00, 0.20),
        "Stellar \n Properties": (-0.15, -0.10),
        "Orbital \n Parameters": (0.10, -0.15),
        "Surface \n Pressure Prior": (-0.10, 0.15),
        "Planet\nPrimary\nProperties": (-0.05, 0.10)
    }
}


# ======================================
# LABEL POSITION OFFSETS
# ======================================

# Additional label-specific positional offsets for clarity
# Format: {habitat_name: {node_name: (x_offset, y_offset)}}
label_offsets = {
    "europa": {
        "Stellar \n Properties": (-0.15, -0.05),
        "Albedo Prior": (-0.20, -0.05),
        "Equilibrium \n Temperature": (0.10, 0.05),
        "Surface \n Pressure Prior": (0.07, -0.05),
        "Methanogens AE v1.0": (0.00, 0.10),
        "Planet\nPrimary\nProperties": (0.00, 0.10)
    },
    "mars": {
        "Equilibrium \n Temperature": (-0.12, 0.02),
        "Planet\nPrimary\nProperties": (0.00, 0.10),
        "Methanogens AE v1.0": (0.00, 0.10),
        "Greenhouse Strength": (0.00, 0.10),
        "Bond Albedo": (0.00, 0.10),
        "Surface \n Pressure Prior": (0.07, -0.05)
    },
        "test_visexoplanet": {
        "Albedo Prior": (-0.05, -0.02),
        "Equilibrium \n Temperature": (0.08, 0.00),
        "Greenhouse \n Effect": (0.00, 0.08),
        "Stellar \n Properties": (-0.10, -0.05),
        "Orbital \n Parameters": (0.10, -0.02),
        "Surface \n Pressure Prior": (-0.08, 0.05),
        "Planet\nPrimary\nProperties": (0.00, 0.12),
        "Cyanobacteria AE v1.0": (0.08, 0.00)  
    }
}
