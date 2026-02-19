// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class FireImpact_Fracture : ModuleRules
{
	public FireImpact_Fracture(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "MeshDescription", "StaticMeshDescription", "RenderCore", "ProceduralMeshComponent", "Sockets", "Networking", "ImageWrapper" });

        PrivateDependencyModuleNames.AddRange(new string[] { "RHI" });
    }
}
