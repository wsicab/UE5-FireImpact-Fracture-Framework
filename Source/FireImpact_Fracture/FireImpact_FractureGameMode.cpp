// Copyright Epic Games, Inc. All Rights Reserved.

#include "FireImpact_FractureGameMode.h"
#include "FireImpact_FractureCharacter.h"
#include "UObject/ConstructorHelpers.h"

AFireImpact_FractureGameMode::AFireImpact_FractureGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

}
