// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"
#include "FireComponent.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"

#include "FractureActor01.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnFireFractureTriggeredSignature, FVector, WorldLocation);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnFractureDataReceivedSignature, int32, NumPieces);

UCLASS()
class FIREIMPACT_FRACTURE_API AFractureActor01 : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AFractureActor01();

protected:

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire System")
    float IgnitionTemperatureMultiplier;

public:
    virtual void Tick(float DeltaTime) override;

 
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Defaults")
    TObjectPtr<UProceduralMeshComponent> FractureMesh;


    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    TArray<FVector> Vertices;

    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    TArray<int32> Triangles;

    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    TArray<FVector> Normals;

    TArray<FVector2D> UV0;
    TArray<FLinearColor> VertexColors;
    TArray<FColor> VertexColors2;
    TArray<FProcMeshTangent> Tangents;


    // mesh
    UFUNCTION(BlueprintCallable)
    void GenerateMesh();

    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    TArray<FVector> NewVertices;

    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    TArray<int32> NewTriangles;

    int Index_A, Index_B, Index_C;
    void Subdivide(int a, int b, int c);


    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Defaults")
    int Recursions;

    TArray<FVector> VertexDictionary;
    TArray<int> IndicesDictionary;
    int i_a, i_b, i_c, i_ab, i_bc, i_ca;


    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Fire System")
    TObjectPtr<UFireComponent> FireComponent;

    UFUNCTION()
    void OnVertexBurnedOut(int32 VertexIndex, FVector WorldLocation);

    UFUNCTION(BlueprintCallable, Category = "Fire System")
    void UpdateFireComponentMesh();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire System")
    TArray<int32> InitialIgnitionIndices;

    UFUNCTION(BlueprintCallable, Category = "Fire System")
    void IgniteAtIndices(const TArray<int32>& IndicesToIgnite);

    UPROPERTY(BlueprintReadWrite, Category = "Fire System")
    TArray<FVector> SavedFireLocations;

    UPROPERTY(BlueprintReadWrite, Category = "Fire System")
    bool bPreserveFireState = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fracture Physics")
    bool bEnablePhysicsOnFracture = false;  

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fracture Physics")
    float PhysicsActivationDelay = 2.0f;  


    UPROPERTY(BlueprintAssignable, Category = "Fire System")
    FOnFireFractureTriggeredSignature OnFireFractureTriggered;

    UPROPERTY(BlueprintAssignable, Category = "Fire System")
    FOnFractureDataReceivedSignature OnFractureDataReceived;


    UFUNCTION(BlueprintCallable, Category = "Networking")
    void BP_OnFractureComplete(int32 NumPieces);

    // fracture data
    TArray<TArray<FVector>> FracturedVertices;
    TArray<TArray<int32>> FracturedTriangles;
    int32 InNumPieces = 0;

    bool bFractureInProgress = false;



private:

    //timer handle
    FTimerHandle InitFireTimerHandle;
    FTimerHandle ReFireTimerHandle;
    FTimerHandle PhysicsTimerHandle;


public:

    bool bIsInitialized;

};