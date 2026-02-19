// Fill out your copyright notice in the Description page of Project Settings.


#include "FireComponent.h"
#include "ProceduralMeshComponent.h"
#include "Particles/ParticleSystemComponent.h"
#include "Kismet/GameplayStatics.h" 
#include "UObject/ConstructorHelpers.h"

// Sets default values for this component's properties
UFireComponent::UFireComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.05f;          //20fps
    PrimaryComponentTick.bStartWithTickEnabled = true;

    // default setting
    HeatDiffusionRate = 2.5f;
    IgnitionTemperature = 190.0f;
    BurningTemperature = 250.0f;
    MaxBurnDuration = 0.2f;
    AmbientTemperature = 20.0f;
    HeatGenerationRate = 200.0f;
    // particles 
    ParticleSamplingRate = 0.15f;  
    SmokeParticleScale = 0.3f;      
    FireParticleScale = 0.4f;       

    bIsInitialized = false;

    FireParticleTemplate = nullptr;
    SmokeParticleTemplate = nullptr;

}


// Called when the game starts
void UFireComponent::BeginPlay()
{
    Super::BeginPlay();


}

void UFireComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);

    TArray<int32> IndicesToRemove;

    for (auto& ParticlePair : VertexParticles)
    {
        bool bRemoved = false;

        // Smoke 
        if (ParticlePair.Value.SmokeParticle)
        {
            if (IsValid(ParticlePair.Value.SmokeParticle))
            {
                ParticlePair.Value.SmokeParticle->DeactivateSystem();
            }
            ParticlePair.Value.SmokeParticle = nullptr;
            bRemoved = true;
        }

        // Fire 
        if (IsValid(ParticlePair.Value.FireParticle))
        {
            ParticlePair.Value.FireParticle->DeactivateSystem();
            ParticlePair.Value.FireParticle = nullptr;
            bRemoved = true;
        }

        if (bRemoved)
        {
            IndicesToRemove.Add(ParticlePair.Key);
        }
    }

    for (int32 Index : IndicesToRemove)
    {
        VertexParticles.Remove(Index);
    }

    UE_LOG(LogTemp, Warning, TEXT("FireComponent: EndPlay complete (%d particles)"),
        IndicesToRemove.Num());
}

void UFireComponent::SetIgnitionPoint(int32 VertexIndex, float InitialTemperature)
{
    if (!bIsInitialized)
    {
        UE_LOG(LogTemp, Error, TEXT("FireComponent::SetIgnitionPoint!"));
        return;
    }

    if (FFireVertexTemperatureData2* TempData = VertexTemperature.Find(VertexIndex))
    {
        TempData->Temperature = InitialTemperature;

        // set temperature
        if (InitialTemperature >= BurningTemperature)
        {
            TempData->CombustionState = EFireCombustionState2::Burning;
            TempData->BurnDuration = 0.0f;

        }
        else if (InitialTemperature >= IgnitionTemperature)
        {
            TempData->CombustionState = EFireCombustionState2::Igniting;

        }
        else
        {
            TempData->CombustionState = EFireCombustionState2::Heating;

        }

        TempData->FuelAmount = 100.0f;
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("FireComponent::SetIgnitionPoint - Vertex %d null"), VertexIndex);
    }

}

void UFireComponent::ResetAllTemperatures()
{
    for (auto& TempPair : VertexTemperature)
    {
        TempPair.Value.Temperature = AmbientTemperature;
        TempPair.Value.CombustionState = EFireCombustionState2::Unburned;
        TempPair.Value.BurnDuration = 0.0f;
        TempPair.Value.FuelAmount = 100.0f;
    }

    for (auto& WeightPair : VertexWeights)
    {
        WeightPair.Value = 1.0f;
    }

    BurnedOutNotifiedVertices.Empty();
    bBurnedOutTriggeredThisTick = false;

    // 안전한 파티클 정리
    for (auto& ParticlePair : VertexParticles)
    {
        if (ParticlePair.Value.SmokeParticle && IsValid(ParticlePair.Value.SmokeParticle))
        {
            ParticlePair.Value.SmokeParticle->DeactivateSystem();
        }
        if (ParticlePair.Value.FireParticle && IsValid(ParticlePair.Value.FireParticle))
        {
            ParticlePair.Value.FireParticle->DeactivateSystem();
        }
    }
    VertexParticles.Empty();

    UE_LOG(LogTemp, Warning, TEXT("FireComponent: reset"));
}

void UFireComponent::ResetTemperaturesOnly()
{
    // 온도와 가중치만 리셋, 파티클은 유지
    for (auto& TempPair : VertexTemperature)
    {
        TempPair.Value.Temperature = AmbientTemperature;
        TempPair.Value.CombustionState = EFireCombustionState2::Unburned;
        TempPair.Value.BurnDuration = 0.0f;
        TempPair.Value.FuelAmount = 100.0f;
    }

    for (auto& WeightPair : VertexWeights)
    {
        WeightPair.Value = 1.0f;
    }

    BurnedOutNotifiedVertices.Empty();
    bBurnedOutTriggeredThisTick = false;

}

void UFireComponent::InitializeFireSystem()
{
    if (bIsInitialized)
    {
        return;
    }

    if (!TargetMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("FireComponent: TargetMeshComponent null!"));
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("FireComponent::InitializeFireSystem() start"));

    BuildAdjacencyCache();

    FProcMeshSection* Section = TargetMeshComponent->GetProcMeshSection(0);
    if (Section)
    {
        int32 NumVertices = Section->ProcVertexBuffer.Num();

        VertexTemperature.Empty();
        VertexWeights.Empty();

        for (int32 i = 0; i < NumVertices; i++)
        {
            VertexTemperature.Add(i, FFireVertexTemperatureData2());
            VertexWeights.Add(i, 1.0f);
        }

        bIsInitialized = true;
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("FireComponent: ProcMeshSection null!"));
    }

}

void UFireComponent::ForceReinitialize()
{
    bIsInitialized = false;

    VertexTemperature.Empty();
    VertexWeights.Empty();
    AdjacencyCache.Empty();

    BurnedOutNotifiedVertices.Empty();
    bBurnedOutTriggeredThisTick = false;

    for (auto& ParticlePair : VertexParticles)
    {
        if (ParticlePair.Value.SmokeParticle && IsValid(ParticlePair.Value.SmokeParticle))
        {
            ParticlePair.Value.SmokeParticle->DeactivateSystem();
        }
        if (ParticlePair.Value.FireParticle && IsValid(ParticlePair.Value.FireParticle))
        {
            ParticlePair.Value.FireParticle->DeactivateSystem();
        }
    }
    VertexParticles.Empty();

    InitializeFireSystem();

}


// Called every frame
void UFireComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!bIsInitialized)
        return;

    bBurnedOutTriggeredThisTick = false;

    UpdateTemperatureDiffusion(DeltaTime);
    UpdateCombustionStates(DeltaTime);
    UpdateVertexColors();

}

void UFireComponent::UpdateTemperatureDiffusion(float DeltaTime)
{
    TMap<int32, float> TemperatureChanges;

    for (auto& TempPair : VertexTemperature)
    {
        int32 VertexIndex = TempPair.Key;
        FFireVertexTemperatureData2& TempData = TempPair.Value;
        float NewTemp = TempData.Temperature;

        // generate heat
        if (TempData.CombustionState == EFireCombustionState2::Burning && TempData.FuelAmount > 0.0f)
        {
            float HeatGeneration = HeatGenerationRate * DeltaTime;
            float FuelRatio = FMath::Clamp(TempData.FuelAmount / 100.0f, 0.3f, 1.0f);
            NewTemp += HeatGeneration * FuelRatio;
            NewTemp = FMath::Min(NewTemp, BurningTemperature * 1.5f);
        }

        // heat diffusion
        TArray<int32> AdjacentIndices;
        AdjacencyCache.MultiFind(VertexIndex, AdjacentIndices);

        float TotalHeatTransfer = 0.0f;
        int32 ValidNeighbors = 0;

        for (int32 AdjacentIndex : AdjacentIndices)
        {
            if (FFireVertexTemperatureData2* AdjacentTempData = VertexTemperature.Find(AdjacentIndex))
            {
                float TempDifference = AdjacentTempData->Temperature - TempData.Temperature;
                float Multiplier = 1.0f;

                if (AdjacentTempData->CombustionState == EFireCombustionState2::Burning && AdjacentTempData->FuelAmount > 0.0f)
                    Multiplier = 3.0f;
                else if (AdjacentTempData->CombustionState == EFireCombustionState2::Igniting)
                    Multiplier = 2.0f;

                TotalHeatTransfer += TempDifference * Multiplier;
                ValidNeighbors++;
            }
        }

        if (ValidNeighbors > 0)
        {
            float HeatDiffusion = (TotalHeatTransfer / ValidNeighbors) * HeatDiffusionRate * DeltaTime;
            NewTemp += HeatDiffusion;
        }

        // cooling
        if (TempData.CombustionState == EFireCombustionState2::Unburned || TempData.CombustionState == EFireCombustionState2::BurnedOut)
        {
            NewTemp -= (NewTemp - AmbientTemperature) * CoolingRate * DeltaTime;
        }

        NewTemp = FMath::Max(NewTemp, AmbientTemperature);
        TemperatureChanges.Add(VertexIndex, NewTemp);
    }

    // temperature update
    for (const auto& TempChange : TemperatureChanges)
    {
        if (FFireVertexTemperatureData2* TempData = VertexTemperature.Find(TempChange.Key))
        {
            TempData->Temperature = TempChange.Value;
        }
    }

}

void UFireComponent::UpdateCombustionStates(float DeltaTime)
{
    static bool bGlobalBurnedOutTriggered = false;

    for (auto& TempPair : VertexTemperature)
    {
        int32 VertexIndex = TempPair.Key;
        FFireVertexTemperatureData2& TempData = TempPair.Value;
        EFireCombustionState2 PreviousState = TempData.CombustionState;

        if (TempData.CombustionState == EFireCombustionState2::Burning)
        {
            TempData.BurnDuration += DeltaTime;
            TempData.FuelAmount -= DeltaTime * 8.0f;
            TempData.FuelAmount = FMath::Max(TempData.FuelAmount, 0.0f);
        }

        switch (TempData.CombustionState)
        {
        case EFireCombustionState2::Unburned:
            if (TempData.Temperature >= IgnitionTemperature * 0.3f && TempData.FuelAmount > 0.5f)
            {
                TempData.CombustionState = EFireCombustionState2::Heating;
            }
            break;

        case EFireCombustionState2::Heating:
            if (TempData.Temperature >= IgnitionTemperature * 0.5f && TempData.FuelAmount > 0.5f)
            {
                TempData.CombustionState = EFireCombustionState2::Igniting;

                if (ShouldSpawnParticleForVertex(VertexIndex))
                {
                    FVector WorldLoc = GetVertexWorldLocation(VertexIndex);
                    if (CanSpawnParticleAtLocation(WorldLoc))
                    {
                        SpawnSmokeParticle(VertexIndex, WorldLoc);
                    }
                }
            }
            break;

        case EFireCombustionState2::Igniting:
            if (TempData.Temperature >= IgnitionTemperature * 0.7f && TempData.FuelAmount > 0.0f)
            {
                TempData.CombustionState = EFireCombustionState2::Burning;
                TempData.BurnDuration = 0.0f;

                if (VertexParticles.Contains(VertexIndex))
                {
                    FVector WorldLoc = GetVertexWorldLocation(VertexIndex);
                    TransitionToFireParticle(VertexIndex, WorldLoc);
                }
            }
            break;

        case EFireCombustionState2::Burning:
            if (TempData.FuelAmount <= 0.1f)// || TempData.BurnDuration >= MaxBurnDuration * 1.2f)
            {
                TempData.CombustionState = EFireCombustionState2::BurnedOut;

  
                if (!BurnedOutNotifiedVertices.Contains(VertexIndex) &&
                    !bBurnedOutTriggeredThisTick &&
                    !bGlobalBurnedOutTriggered)
                {
                    BurnedOutNotifiedVertices.Add(VertexIndex);
                    bBurnedOutTriggeredThisTick = true;
                    bGlobalBurnedOutTriggered = true;

                    FVector WorldLoc = GetVertexWorldLocation(VertexIndex);
                    OnVertexBurnedOut.Broadcast(VertexIndex, WorldLoc);

                    static int32 BurnedOutCount = 0;
                    BurnedOutCount++;
                    return;
                }
            }
            break;
        }

        if (TempData.CombustionState == EFireCombustionState2::BurnedOut)
        {
            if (TempData.Temperature < IgnitionTemperature * 0.5f)
            {
                if (VertexParticles.Contains(VertexIndex) && VertexParticles[VertexIndex].bIsActive)
                {
                    DeactivateParticle(VertexIndex);
                }
            }
        }

        if (float* Weight = VertexWeights.Find(VertexIndex))
        {
            if (TempData.CombustionState == EFireCombustionState2::Burning)
            {
                *Weight -= DeltaTime * 0.03f;
                *Weight = FMath::Max(*Weight, 0.2f);
            }
        }

        if (TempData.CombustionState == EFireCombustionState2::Burning &&
            PreviousState != EFireCombustionState2::Burning)
        {
            float CurrentWeight = VertexWeights.FindRef(VertexIndex);
            OnVertexBurning.Broadcast(VertexIndex, TempData.Temperature, CurrentWeight);
        }
    }
}

bool UFireComponent::CanSpawnParticleAtLocation(const FVector& Location) const
{
    for (const auto& ParticlePair : VertexParticles)
    {
        if (ParticlePair.Value.bIsActive)
        {
            float Distance = FVector::Dist(Location, ParticlePair.Value.WorldLocation);
            if (Distance < ParticleSpawnDistance)
            {
                return false; 
            }
        }
    }
    return true;

}

void UFireComponent::BuildAdjacencyCache()
{
    AdjacencyCache.Empty();

    if (!TargetMeshComponent)
        return;

    FProcMeshSection* Section = TargetMeshComponent->GetProcMeshSection(0);
    if (!Section)
        return;

    for (int32 i = 0; i < Section->ProcIndexBuffer.Num(); i += 3)
    {
        int32 V0 = Section->ProcIndexBuffer[i];
        int32 V1 = Section->ProcIndexBuffer[i + 1];
        int32 V2 = Section->ProcIndexBuffer[i + 2];

        AdjacencyCache.AddUnique(V0, V1);
        AdjacencyCache.AddUnique(V0, V2);
        AdjacencyCache.AddUnique(V1, V0);
        AdjacencyCache.AddUnique(V1, V2);
        AdjacencyCache.AddUnique(V2, V0);
        AdjacencyCache.AddUnique(V2, V1);
    }

}

void UFireComponent::UpdateVertexColors()
{
    if (!TargetMeshComponent)
        return;

    FProcMeshSection* Section = TargetMeshComponent->GetProcMeshSection(0);
    if (!Section)
        return;

    bool bColorChanged = false;
    int32 ChangedCount = 0;  

    for (auto& TempPair : VertexTemperature)
    {
        int32 VertexIndex = TempPair.Key;
        FColor NewColor = GetColorFromTemperature(TempPair.Value);

        if (VertexIndex < Section->ProcVertexBuffer.Num())
        {
            FColor& OldColor = Section->ProcVertexBuffer[VertexIndex].Color;

            if (OldColor != NewColor)
            {
                OldColor = NewColor;
                bColorChanged = true;
                ChangedCount++;
            }
        }
    }

    if (bColorChanged)
    {
        TargetMeshComponent->MarkRenderStateDirty();

    }
}

FColor UFireComponent::GetColorFromTemperature(const FFireVertexTemperatureData2& TempData) const
{
    switch (TempData.CombustionState)
    {
    case EFireCombustionState2::Unburned:
        return FColor::Blue;
    case EFireCombustionState2::Heating:
        return FColor::Green;
    case EFireCombustionState2::Igniting:
        return FColor::Yellow;
    case EFireCombustionState2::Burning:
        return FColor::Red;
    case EFireCombustionState2::BurnedOut:
        return FColor(80, 80, 80);
    default:
        return FColor::White;
    }

}

void UFireComponent::SpawnSmokeParticle(int32 VertexIndex, const FVector& WorldLocation)
{
    // check particles num
    if (ActiveParticleCount >= MaxActiveParticles)
    {
        return;  
    }

    if (!SmokeParticleTemplate)
    {
        static bool bWarningShown = false;
        if (!bWarningShown)
        {
            bWarningShown = true;
        }
        return;
    }

    UParticleSystemComponent* SmokeComp = UGameplayStatics::SpawnEmitterAtLocation(
        GetWorld(),
        SmokeParticleTemplate,
        WorldLocation,
        FRotator::ZeroRotator,
        FVector(SmokeParticleScale),
        true,
        EPSCPoolMethod::AutoRelease,  // None → AutoRelease 
        true
    );

    if (SmokeComp)
    {
        FVertexParticleData& ParticleData = VertexParticles.FindOrAdd(VertexIndex);
        ParticleData.SmokeParticle = SmokeComp;
        ParticleData.WorldLocation = WorldLocation;
        ParticleData.bIsActive = true;

        ActiveParticleCount++;  
    }

}

void UFireComponent::TransitionToFireParticle(int32 VertexIndex, const FVector& WorldLocation)
{
    if (!FireParticleTemplate)
    {
        static bool bWarningShown = false;
        if (!bWarningShown)
        {
            UE_LOG(LogTemp, Warning, TEXT("FireParticleTemplate null."));
            bWarningShown = true;
        }
        return;
    }

    if (FVertexParticleData* ParticleData = VertexParticles.Find(VertexIndex))
    {
        if (ParticleData->SmokeParticle && IsValid(ParticleData->SmokeParticle))
        {
            ParticleData->SmokeParticle->DeactivateSystem();
            ParticleData->SmokeParticle->DestroyComponent();
        }
    }

    UParticleSystemComponent* FireComp = UGameplayStatics::SpawnEmitterAtLocation(
        GetWorld(),
        FireParticleTemplate,
        WorldLocation,
        FRotator::ZeroRotator,
        FVector(FireParticleScale),
        true,
        EPSCPoolMethod::None,
        true
    );

    if (FireComp)
    {
        FVertexParticleData& ParticleData = VertexParticles.FindOrAdd(VertexIndex);
        ParticleData.FireParticle = FireComp;
        ParticleData.WorldLocation = WorldLocation;
        ParticleData.bIsActive = true;
    }

}

void UFireComponent::DeactivateParticle(int32 VertexIndex)
{
    if (FVertexParticleData* ParticleData = VertexParticles.Find(VertexIndex))
    {
        if (IsValid(ParticleData->SmokeParticle))
        {
            ParticleData->SmokeParticle->DeactivateSystem();
            ActiveParticleCount--;
        }

        if (IsValid(ParticleData->FireParticle))
        {
            ParticleData->FireParticle->DeactivateSystem();
            ActiveParticleCount--;
        }

        ParticleData->bIsActive = false;
    }

}

FVector UFireComponent::GetVertexWorldLocation(int32 VertexIndex) const
{
    if (!TargetMeshComponent)
        return FVector::ZeroVector;

    FProcMeshSection* Section = TargetMeshComponent->GetProcMeshSection(0);
    if (!Section || VertexIndex >= Section->ProcVertexBuffer.Num())
        return FVector::ZeroVector;

    FVector LocalPos = (FVector)Section->ProcVertexBuffer[VertexIndex].Position;
    return TargetMeshComponent->GetComponentTransform().TransformPosition(LocalPos);

}


bool UFireComponent::ShouldSpawnParticleForVertex(int32 VertexIndex) const
{
    float RandomValue = FMath::FRand();
    return RandomValue < ParticleSamplingRate;

}

void UFireComponent::PauseFireSimulation(bool bPause)
{
    SetComponentTickEnabled(!bPause);

    if (bPause)
    {
        UE_LOG(LogTemp, Warning, TEXT("fire simulation pause"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("fire simulation restart"));
    }
}

TArray<FVector> UFireComponent::SaveFireParticleLocations() const
{
    TArray<FVector> Locations;

    for (const auto& ParticlePair : VertexParticles)
    {
        if (ParticlePair.Value.bIsActive)
        {
            Locations.Add(ParticlePair.Value.WorldLocation);
        }
    }

    return Locations;
}

void UFireComponent::RestoreFireAtLocations(const TArray<FVector>& WorldLocations, float Temperature)
{
    if (!bIsInitialized || !TargetMeshComponent)
    {
        return;
    }

    FProcMeshSection* Section = TargetMeshComponent->GetProcMeshSection(0);
    if (!Section)
    {
        return;
    }

    int32 RestoredCount = 0;

    for (const FVector& WorldLoc : WorldLocations)
    {
        int32 ClosestVertex = -1;
        float MinDistance = FLT_MAX;

        for (int32 i = 0; i < Section->ProcVertexBuffer.Num(); i++)
        {
            FVector VertexWorldLoc = GetVertexWorldLocation(i);
            float Distance = FVector::Dist(VertexWorldLoc, WorldLoc);

            if (Distance < MinDistance)
            {
                MinDistance = Distance;
                ClosestVertex = i;
            }
        }

        if (MinDistance > 50.0f || ClosestVertex == -1)
            continue;

        if (FFireVertexTemperatureData2* TempData = VertexTemperature.Find(ClosestVertex))
        {
            TempData->Temperature = Temperature;
            TempData->CombustionState = EFireCombustionState2::Burning;
            TempData->BurnDuration = 0.0f;
            TempData->FuelAmount = 100.0f;

            FVector VertexWorldLoc = GetVertexWorldLocation(ClosestVertex);

            if (!VertexParticles.Contains(ClosestVertex))
            {
                if (ShouldSpawnParticleForVertex(ClosestVertex) &&
                    CanSpawnParticleAtLocation(VertexWorldLoc))
                {
                    TransitionToFireParticle(ClosestVertex, VertexWorldLoc);
                    RestoredCount++;
                }
            }
        }
    }
}