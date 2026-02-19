// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Particles/ParticleSystemComponent.h"
#include "FireComponent.generated.h"

// Forward Declarations
class UParticleSystem;
class UProceduralMeshComponent;

UENUM(BlueprintType)
enum class EFireCombustionState2 : uint8
{
    Unburned    UMETA(DisplayName = "Unburned"),
    Heating     UMETA(DisplayName = "Heating"),
    Igniting    UMETA(DisplayName = "Igniting"),
    Burning     UMETA(DisplayName = "Burning"),
    BurnedOut   UMETA(DisplayName = "Burned Out")
};

USTRUCT(BlueprintType)
struct FFireVertexTemperatureData2
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float Temperature;

    UPROPERTY(BlueprintReadOnly)
    EFireCombustionState2 CombustionState;

    UPROPERTY(BlueprintReadOnly)
    float BurnDuration;

    UPROPERTY(BlueprintReadOnly)
    float FuelAmount;

    FFireVertexTemperatureData2()
        : Temperature(20.0f)
        , CombustionState(EFireCombustionState2::Unburned)
        , BurnDuration(0.0f)
        , FuelAmount(100.0f)
    {
    }
};

USTRUCT()
struct FVertexParticleData
{
    GENERATED_BODY()

    UPROPERTY()
    TObjectPtr<UParticleSystemComponent> SmokeParticle;
    UPROPERTY()
    TObjectPtr<UParticleSystemComponent> FireParticle;

    UPROPERTY()
    FVector WorldLocation;

    UPROPERTY()
    bool bIsActive;


    FVertexParticleData()
    {
        SmokeParticle = nullptr;
        FireParticle = nullptr;
        WorldLocation = FVector::ZeroVector;
        bIsActive = false;
    }
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnVertexBurnedOutSignature, int32, VertexIndex, FVector, WorldLocation);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnVertexBurningSignature2, int32, VertexIndex, float, Temperature, float, DamageWeight);



UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class FIREIMPACT_FRACTURE_API UFireComponent : public UActorComponent
{
	GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    UFireComponent();

    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    // 델리게이트 
    UPROPERTY(BlueprintAssignable, Category = "Fire Events")
    FOnVertexBurningSignature2 OnVertexBurning;

    UPROPERTY(BlueprintAssignable, Category = "Fire Events")
    FOnVertexBurnedOutSignature OnVertexBurnedOut;

    // 화재 시작 
    UFUNCTION(BlueprintCallable, Category = "Fire")
    void SetIgnitionPoint(int32 VertexIndex, float InitialTemperature = 500.0f);

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void ResetAllTemperatures();

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void ResetTemperaturesOnly(); 

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void InitializeFireSystem();

 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float HeatDiffusionRate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float IgnitionTemperature;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float BurningTemperature;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float MaxBurnDuration;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float AmbientTemperature;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float HeatGenerationRate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temperature System")
    float CoolingRate = 0.15f;  
    //파티클 시스템 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    UParticleSystem* FireParticleTemplate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    UParticleSystem* SmokeParticleTemplate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    float ParticleSpawnDistance = 100.0f;  

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    float ParticleSamplingRate = 0.05f;   

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    int32 MaxActiveParticles = 20;

    // 메시 참조
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    UProceduralMeshComponent* TargetMeshComponent;

    // 가중치 
    UFUNCTION(BlueprintCallable, Category = "Fire")
    TMap<int32, float> GetVertexWeights() const { return VertexWeights; }

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void ForceReinitialize();

    // 파티클 스케일
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    float SmokeParticleScale = 0.3f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle System")
    float FireParticleScale = 0.5f;

    // 화재 상태 저장/복원
    UFUNCTION(BlueprintCallable, Category = "Fire")
    TArray<FVector> SaveFireParticleLocations() const;

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void RestoreFireAtLocations(const TArray<FVector>& WorldLocations, float Temperature);

private:
    bool bIsInitialized;

    TMap<int32, FFireVertexTemperatureData2> VertexTemperature;

    TMap<int32, float> VertexWeights;

    TMultiMap<int32, int32> AdjacencyCache;

    TMap<int32, FVertexParticleData> VertexParticles;

    void UpdateTemperatureDiffusion(float DeltaTime);
    void UpdateCombustionStates(float DeltaTime);
    void BuildAdjacencyCache();
    void UpdateVertexColors();
    FColor GetColorFromTemperature(const FFireVertexTemperatureData2& TempData) const;


    void SpawnSmokeParticle(int32 VertexIndex, const FVector& WorldLocation);
    void TransitionToFireParticle(int32 VertexIndex, const FVector& WorldLocation);
    void DeactivateParticle(int32 VertexIndex);
    FVector GetVertexWorldLocation(int32 VertexIndex) const;


    bool CanSpawnParticleAtLocation(const FVector& Location) const;
    bool ShouldSpawnParticleForVertex(int32 VertexIndex) const;

    int32 ActiveParticleCount = 0;

public: 
    UFUNCTION(BlueprintCallable, Category = "Fire")
    int32 GetVertexCount() const { return VertexTemperature.Num(); }
public:
    UFUNCTION(BlueprintCallable, Category = "Fire")
    void PauseFireSimulation(bool bPause);

private:
    //bool bIsInitialized;

    TSet<int32> BurnedOutNotifiedVertices; 
    bool bBurnedOutTriggeredThisTick;       



};